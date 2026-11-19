#!/usr/bin/env python3
"""M4.45 — READ-ONLY PPO / source / resource fidelity audit for G1Joystick.

Reads the already finished M4.30 training archive and (if locally available)
its pinned source files. Does not import JAX, MuJoCo or open any checkpoint for
training; does not mutate source or model. Outputs JSON and Markdown.

Usage:
  python m445_paper_fidelity_audit.py \
    --archive m430_seed1.tar.gz \
    --source-repo /home/labrobot/nguyenbui/humanoid/G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/source/mujoco_playground_2025

References: Demonstrating MuJoCo Playground, RSS 2025, Appendix F.2,
Table XV (default locomotion Brax PPO) + Table XX (G1Joystick overrides).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PIN = '9bcbe736177472ba35d60773e3030a36e37962b7'
JOYSTICK_SHA = '9d3b6d04fbc86c99f1ebf69bb7e3fc7ae11dbd8588ec3fd2daf4d8e28a41a5a4'
FINAL_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
REPORT = 'm430_seed1/pilot_20261114_120515_1722937_report.json'
MANIFEST = 'm430_seed1/m430_manifest.json'
FINISHED = 'm430_seed1/m430_run_finished.json'
TRAIN_SCRIPT = 'm430_seed1/pilot_seed1.py'
JOYSTICK_REL = 'mujoco_playground/_src/locomotion/g1/joystick.py'
RANDOMIZER_REL = 'mujoco_playground/_src/locomotion/g1/randomize.py'
LOC_PARAMS_REL = 'mujoco_playground/config/locomotion_params.py'
PAPER = {
    'episode_length':1000,             # Environment default; not a Table XX field
    'num_timesteps': 400000000,
    'num_evals': 16,
    'num_resets_per_eval': 1,
    'reward_scaling': .1,
    'normalize_observations': True,  # Table XV inherited locomotion default
    'action_repeat': 1,              # Table XV inherited locomotion default
    'unroll_length': 32,
    'num_minibatches': 32,           # Table XV inherited locomotion default
    'num_updates_per_batch': 5,
    'discounting': .98,
    'learning_rate': .0001,
    'entropy_cost': 0.,
    'num_envs': 32768,
    'batch_size': 1024,
    'clipping_epsilon': .2,
    'max_grad_norm': 1.,             # Table XV inherited locomotion default
    'network_factory': {
        'policy_hidden_layer_sizes': [512,256,64],
        'value_hidden_layer_sizes': [256,256,256,256],
        'policy_obs_key':'state',
        'value_obs_key':'privileged_state',
    },
}
G1_TABLE_XX_FIELDS = {
    'num_timesteps','num_evals','num_resets_per_eval','reward_scaling',
    'unroll_length','num_updates_per_batch','discounting','learning_rate',
    'entropy_cost','num_envs','batch_size','clipping_epsilon',
    'policy_hidden_layer_sizes','value_hidden_layer_sizes','value_obs_key',
}


def file_sha(path):
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_tar(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        names={m.name for m in tar.getmembers() if m.isfile()}
        missing={REPORT,MANIFEST,FINISHED,TRAIN_SCRIPT}-names
        if missing:raise RuntimeError(f'M445_ARCHIVE_MISSING_REQUIRED_FILES: {missing}')
        def read(name):
            f=tar.extractfile(name)
            if f is None:raise RuntimeError('M445_TAR_EXTRACT_FAILED '+name)
            return f.read()
        report=json.loads(read(REPORT))
        manifest=json.loads(read(MANIFEST))
        finished=json.loads(read(FINISHED))
        source_bytes=read(TRAIN_SCRIPT)
        final_name=next((Path(x['file']).name for x in report.get('checkpoints',[])
                         if x.get('step')==401080320),None)
        final_rel='m430_seed1/'+final_name if final_name else None
        final_digest=None
        if final_rel in names:
            final_digest=hashlib.sha256(read(final_rel)).hexdigest()
    return report,manifest,finished,source_bytes,final_digest


def compare_values(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        return math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-10)
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(compare_values(a[k],b[k]) for k in a)
    return a==b


def schedule(cfg):
    rollout_steps=int(cfg['batch_size']*cfg['num_minibatches']*cfg['unroll_length']*cfg['action_repeat'])
    assert cfg['batch_size'] * cfg['num_minibatches']==cfg['num_envs']
    intervals=int(cfg['num_evals'])-1
    rollout_batches=math.ceil(cfg['num_timesteps']/(intervals*rollout_steps))*intervals
    return {'environment_steps_per_rollout_batch':rollout_steps,
            'rollout_batches':rollout_batches,
            'actual_scheduled_environment_steps':rollout_batches*rollout_steps,
            'optimizer_minibatch_updates':rollout_batches*cfg['num_minibatches']*cfg['num_updates_per_batch'],
            'evaluation_intervals':intervals}


def source_observations(repo):
    results={'checked_direct_local_pinned_repo':False, 'findings':{},'unresolved':[]}
    if repo is None or not repo.is_dir():
        results['unresolved'].append('Pinned Git repo is not mounted here; source conclusions are limited to prior audited extracts and training script.')
        return results
    head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
    if head.returncode!=0:raise RuntimeError('M445_SOURCE_GIT_REV_PARSE_FAILED')
    commit=head.stdout.strip()
    if commit!=PIN:raise RuntimeError('M445_SOURCE_COMMIT_MISMATCH: '+commit)
    stat=subprocess.run(['git','-C',str(repo),'status','--porcelain','--',JOYSTICK_REL,RANDOMIZER_REL],capture_output=True,text=True)
    if stat.returncode!=0 or stat.stdout.strip():raise RuntimeError('M445_PINNED_SOURCE_FILES_MODIFIED')
    joy=repo/JOYSTICK_REL; rand=repo/RANDOMIZER_REL
    if not joy.is_file():raise RuntimeError('M445_PINNED_JOYSTICK_MISSING')
    if file_sha(joy)!=JOYSTICK_SHA:raise RuntimeError('M445_JOYSTICK_SHA_MISMATCH')
    src=joy.read_text(encoding='utf-8')
    ast.parse(src)
    obs={
      'uses_knees_bent_keyframe': 'keyframe("knees_bent")' in src,
      'default_push_enabled':bool(re.search(r'push_config=config_dict.create\(\s*enable=True',src)),
      'default_push_magnitude_range_0_1_to_2_0': 'magnitude_range=[0.1, 2.0]' in src,
      'default_reward_tracking_lin_vel_1_0':bool(re.search(r'tracking_lin_vel=1\.0',src)),
      'default_reward_tracking_ang_vel_0_75':bool(re.search(r'tracking_ang_vel=0\.75',src)),
      'default_reward_feet_phase_1_0':'feet_phase=1.0' in src,
      'noise_config_present':'noise_config=config_dict.create(' in src,
      'command_config_present':'command_config=config_dict.create(' in src,
      'phase_update_present':'phase_tp1 = state.info["phase"] + state.info["phase_dt"]' in src,
    }
    if not all(obs.values()):raise RuntimeError(f'M445_SOURCE_OBSERVATION_MISMATCH: {obs}')
    results.update({'checked_direct_local_pinned_repo':True, 'commit':commit,'joystick_sha256':file_sha(joy),'findings':obs})
    if rand.is_file():
        rst=rand.read_text(encoding='utf-8'); ast.parse(rst)
        results['randomize_source_sha256']=file_sha(rand)
        # Record evidence, do not assert distribution/paper equivalence from tokens.
        results['randomize_function_defined']=bool(re.search(r'^def\s+domain_randomize\(',rst,re.M))
        results['randomize_source_mentions']={k:(k in rst) for k in ['pair_friction','body_mass','body_ipos','dof_damping','geom_friction','actuator_gainprm']}
        if not results['randomize_function_defined']:
            results['unresolved'].append('Expected domain_randomize() function not found in pinned source.')
    else:
        results['unresolved'].append('Pinned randomize.py missing; DR implementation remains unverified.')
    return results


def build(archive,repo):
    report,manifest,finished,train_bytes,final_digest=read_tar(archive)
    if report.get('status')!='PILOT_COMPLETED' or finished.get('status')!='M430_TRAINING_COMPLETED':
        raise RuntimeError('M445_TRAINING_REPORT_NOT_FINISHED')
    if finished.get('seed')!=1 or manifest.get('seed')!=1:
        raise RuntimeError('M445_TRAIN_SEED_PROVENANCE_MISMATCH')
    if hashlib.sha256(train_bytes).hexdigest()!=manifest['seed1_source_sha256']:
        raise RuntimeError('M445_TRAINING_SCRIPT_SHA_MISMATCH')
    ast.parse(train_bytes.decode('utf-8'))
    script=train_bytes.decode('utf-8')
    if 'cfg.update(seed=1, randomization_fn=randomizer)' not in script:
        raise RuntimeError('M445_TRAINING_RANDOMIZER_NOT_REGISTERED')
    if 'locomotion.get_default_config(ENV)' not in script:
        raise RuntimeError('M445_ORIGINAL_ENV_CONFIG_NOT_LOADED')
    if 'domain_randomization_enabled' not in report or not report['domain_randomization_enabled']:
        raise RuntimeError('M445_DR_REPORT_FLAG_MISSING')
    if report['environment']!='G1JoystickFlatTerrain':
        raise RuntimeError('M445_WRONG_ENVIRONMENT')
    actual=report['scaled_config']; paper_saved=report['paper_config']
    if not compare_values(PAPER,paper_saved):
        raise RuntimeError('M445_SAVED_PAPER_TABLE_CONFIG_DIFFERS_FROM_EXPECTED')
    for key in PAPER:
        if key not in actual:raise RuntimeError('M445_MISSING_ACTUAL_CONFIG_FIELD: '+key)
    if report['checkpoints'][-1]['step']!=401080320 or report['checkpoints'][-1]['sha256']!=FINAL_SHA:
        raise RuntimeError('M445_FINAL_CHECKPOINT_PROVENANCE_MISMATCH')
    if final_digest is not None and final_digest!=FINAL_SHA:
        raise RuntimeError('M445_FINAL_CHECKPOINT_BYTES_SHA_MISMATCH')
    if not report['final_checkpoint_comparison']['tensor_values_equal']:
        raise RuntimeError('M445_FINAL_CHECKPOINT_TENSORS_NOT_EQUAL')
    differences=[];matches=[]
    for key in PAPER:
        if compare_values(PAPER[key],actual[key]): matches.append(key)
        else: differences.append({'key':key,'paper':PAPER[key],'actual':actual[key], 'classification':'hardware_scaled'})
    assert {d['key'] for d in differences}=={'num_envs','batch_size'},differences
    if actual['num_eval_envs']!=2:raise RuntimeError('M445_EVAL_ENV_COUNT_CHANGED')
    sp=schedule(PAPER);sa=schedule(actual)
    if sa['actual_scheduled_environment_steps']!=finished['final_step']:
        raise RuntimeError('M445_SCHEDULE_ACTUAL_STEPS_MISMATCH')
    if sp['actual_scheduled_environment_steps']!=408944640 or sp['optimizer_minibatch_updates']!=62400:
        raise RuntimeError('M445_PAPER_SCHEDULE_RECALC_MISMATCH')
    results={
      'status':'M445_PAPER_FIDELITY_AUDIT_COMPLETED',
      'method':'READ_ONLY_ARCHIVE_AND_PINNED_SOURCE_AUDIT_NO_TRAINING_NO_NEW_SEEDS',
      'reference':'Demonstrating MuJoCo Playground (RSS 2025), Appendix F.2, Tables XV and XX',
      'reference_url':'https://www.roboticsproceedings.org/rss21/p020.pdf',
      'archive_filename':archive.name,'reported_environment':report['environment'],
      'reported_source_commit':PIN,'checkpoint_step':finished['final_step'],
      'checkpoint_sha256':FINAL_SHA,'train_script_sha256':hashlib.sha256(train_bytes).hexdigest(),
      'final_checkpoint_bytes_sha_verified':final_digest is not None,
      'ppo_paper_target':PAPER,'ppo_scaled_actual':actual,
      'matching_ppo_fields':matches,'differing_ppo_fields':differences,
      'non_table_xx_eval_env_count_actual':actual['num_eval_envs'],
      'reference_num_eval_envs_status':'Not specified in Table XX; should be compared to original training runner/default and paper evaluation procedure',
      'in_run_eval_count_provenance':len(report.get('progress',[])),
      'paper_equivalent_schedule_if_full_config':sp,'scaled_schedule_actual':sa,
      'optimizer_update_ratio_scaled_over_full':sa['optimizer_minibatch_updates']/sp['optimizer_minibatch_updates'],
      'source':source_observations(repo),
      'training_script_uses_environment_default_config':True,
      'training_script_passes_domain_randomizer':True,
      'training_script_mutates_reward':'NO_EVIDENCE_IN_CAPTURED_TRAINING_SCRIPT',
      'paper_exact_reproduction':False,
      'limitations':[
        'Table XX and Table XV establish PPO settings; not full equivalence of model, observation, command, reward, domain randomization or evaluation.',
        'Source pinned is from 2025; up-to-date GitHub main may differ and is not a substitute for this training source.',
        'GPU OOM at 32,768 envs previously observed on RTX3060 12GB, so full paper batch was not completed on that hardware.',
        'The effect of extra optimizer updates on gait or push recovery is a hypothesis, not established causation.',
        'The in-training scalar eval reward does not prove identical metric or domain randomization settings to the physical results in the paper.',
        'No policy was retrained or evaluated as part of this audit.'
      ],
      'next_step':'M4.46: compare existing visual/gait metrics against paper and inspect command/reward/domain-randomizer semantics; train only after deciding compute and experiment protocol.'
    }
    return results


def md_report(data):
    a=data['scaled_schedule_actual']; p=data['paper_equivalent_schedule_if_full_config']
    lines=[
      '# M4.45 — Paper Fidelity Audit: Unitree G1 PPO',
      '',
      '**Scope:** read-only evidence audit; no new seeds, no training, no source or checkpoint change.',
      '',
      '## Verdict',
      '',
      '**Hardware-scaled, partially paper-aligned PPO reproduction. NOT paper-exact.**',
      '',
      'Training is complete and the seed-1 checkpoint at step 401,080,320 was verified, but the actual PPO learning schedule differs from the paper-target schedule because the rollout batch and environment count were reduced by four.',
      '',
      '## Source hierarchy',
      '',
      '- **Paper:** RSS 2025, Appendix F.2 Table XV (default locomotion PPO) and Table XX (G1Joystick), '+data['reference_url'],
      '- **Actual train run:** m430_seed1 pilot report, manifest, completed marker and code embedded in the supplied `m430_seed1.tar.gz`.',
      '- **Source:** pinned MuJoCo Playground Git commit `'+PIN+'`'+(' (directly verified in local checkout)' if data['source']['checked_direct_local_pinned_repo'] else ' (source content not directly mounted for this audit run; known source is from earlier source extracts)')+'.',
      '',
      '## PPO configuration',
      '',
      '| Parameter | Paper target | Actual scaled run | Conclusion |',
      '|---|---:|---:|---|',
    ]
    for key in ('episode_length','num_timesteps','num_evals','num_resets_per_eval','reward_scaling','normalize_observations','action_repeat','unroll_length','num_minibatches','num_updates_per_batch','discounting','learning_rate','entropy_cost','num_envs','batch_size','clipping_epsilon','max_grad_norm'):
        pv=data['ppo_paper_target'][key]; av=data['ppo_scaled_actual'][key]
        lines.append(f'| `{key}` | {pv} | {av} | '+('MATCH' if compare_values(pv,av) else '**DIFF (hardware-scaled)**')+' |')
    lines += ['', 'Actor network `(512,256,64)`, critic `(256,256,256,256)`, policy observation key `state` and critic observation key `privileged_state` match target (Table XX + Table XV inheritance).',
      '',
      '## PPO training schedule implications',
      '',
      '| Metric | Paper config schedule (same Brax implementation) | Actual M4.30 run |',
      '|---|---:|---:|',
      f"| Steps / rollout batch | {p['environment_steps_per_rollout_batch']:,} | {a['environment_steps_per_rollout_batch']:,} |",
      f"| Rollout batches | {p['rollout_batches']:,} | {a['rollout_batches']:,} |",
      f"| Scheduled environment steps | {p['actual_scheduled_environment_steps']:,} | {a['actual_scheduled_environment_steps']:,} |",
      f"| Optimizer minibatch updates | {p['optimizer_minibatch_updates']:,} | {a['optimizer_minibatch_updates']:,} |",
      '',
      f"The scaled training uses **{data['optimizer_update_ratio_scaled_over_full']:.3f}×** as many optimizer minibatch updates for approximately the same requested 400M environment steps. This is a verified *algorithmic schedule difference*, not a proven cause of gait or robustness behavior.",
      '',
      '## Environment, reward, randomization and evaluation',
      '',
      '- Actual environment: `G1JoystickFlatTerrain` (registered 2025 code). Paper Table XX labels it `G1Joystick`; exact branch-equivalence for every experiment is **NOT established**.',
      '- Training script creates environments from `locomotion.get_default_config(ENV)` and passes `registry.get_domain_randomizer(ENV)` as `randomization_fn`; the report flags domain randomization enabled. There is no evidence of changing reward scales in that training script; this does not independently prove every source reward/observation function matches the 2025 published experiment.',
      '- Pinned G1 source previously audited: reward categories include linear/angular tracking, feet phase/airtime/slip, pose/joint terms, contact and termination; gait phase continues to advance at zero command; default init pose is `knees_bent`. This is intentionally source behavior and was not changed.',
      '- Pinned source includes timed random world-XY velocity pushes (`0.1–2.0 m/s` source default); diagnostic fixed commands/magnitudes in M4.35–M4.44 are **not** paper evaluation.',
      '- Actual `num_eval_envs=2` is a hardware-driven choice. **Table XX does not specify this value**; `2` means internal training eval has high statistical uncertainty. Earlier full configuration preflight showed training-runner default `128`; this needs proper evaluation-equivalence classification, not automatic Table XX mismatch.',
      '',
      '### Resource limitation already demonstrated',
      '',
      'M4.22E attempted 32,768 environments on **RTX 3060 12 GB** and failed with `RESOURCE_EXHAUSTED: Out of memory` before completing training. Increasing the environment count to 32,768 without a new memory plan or larger GPU is not currently viable.',
      '',
      '## Priorities and decision gate',
      '',
      '1. **M4.46 (no training, no new seed):** audit exact source reward, command sampling, G1 randomizer and observation semantics against paper Section C; summarize existing gait evidence and videos. Identify which apparent gait deviations are actually source-defined versus learned-policy behavior.',
      '2. **M4.47 (design decision):** pick either (A) a higher-memory GPU to attempt full 32,768-env Paper PPO, or (B) an explicitly labeled memory-scaled reproduction. Avoid treating changed rollout batch / optimizer update schedule as identical PPO.',
      '3. **M4.48 (only when justified):** train a controlled new run using the committed protocol, compare to the retained 401M baseline, and report both natural gait and push robustness. Do not adjust reward, PD, termination or phase without separate experimental objective and paper justification.',
      '',
      '## Audit execution',
      '',
      '- Report and manifest decoded, seed and source-file hashes checked.',
      '- Callback final checkpoint SHA-256: `'+data['checkpoint_sha256']+'`'+(' (verified against checkpoint bytes in supplied archive)' if data['final_checkpoint_bytes_sha_verified'] else ' (verified against training manifest only)')+'.',
      '- Runtime local pinned-source verification: '+('PASS' if data['source']['checked_direct_local_pinned_repo'] else 'NOT AVAILABLE in this audit environment; invoke script on actual Ubuntu source for direct proof')+'.',
      '- No JAX/MJX simulation or seed runs performed.',
      '',
    ]
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('m430_seed1.tar.gz'))
    parser.add_argument('--source-repo',type=Path,default=None)
    parser.add_argument('--output-dir',type=Path,default=Path('m445_paper_fidelity'))
    args=parser.parse_args()
    if not args.archive.is_file():
        parser.error('Missing existing M4.30 archive: '+str(args.archive))
    report=build(args.archive,args.source_repo)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output_json=args.output_dir/'M445_AUDIT_RESULTS.json'
    output_md=args.output_dir/'M445_PAPER_FIDELITY_REPORT.md'
    if output_json.exists() or output_md.exists():
        parser.error('Refusing overwrite. Provide an empty --output-dir or review previous outputs.')
    output_json.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    output_md.write_text(md_report(report),encoding='utf-8')
    print('M445_PAPER_FIDELITY_AUDIT_COMPLETED')
    print('MATCHING_PPO_FIELDS:',len(report['matching_ppo_fields']))
    print('DIFF_PPO_FIELDS:',report['differing_ppo_fields'])
    print('OPTIMIZER_UPDATES_RATIO:',round(report['optimizer_update_ratio_scaled_over_full'],4))
    print('DIRECT_PINNED_SOURCE_CHECK:',report['source']['checked_direct_local_pinned_repo'])
    print('REPORT:',output_md)
    print('JSON:',output_json)

if __name__=='__main__':
    main()
