import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tsbench.schema import Observation, CLASSES, uniform_prediction, validate_prediction, longest_run
from tsbench.generate import generate_case,generate_split,validate_config,public_record,digest
from tsbench.tools import ToolSession,ToolError,filled,trend_stats,spectrum_stats
from tsbench.agents import run_method,verified_from,OllamaController,METHODS,verify
from tsbench.metrics import fit_temperature,temperature_scale,summarize,paired_bootstrap,task_scores,ece
from tsbench.cli import run,semantic_records

ROOT=Path(__file__).resolve().parents[1]
def config():return json.loads((ROOT/'configs/smoke.json').read_text())
def record(pred=None,failed=False,split='calibration',id='a'):
    return {'id':id,'family':'linear_sine','split':split,
            'labels':{'trend':'flat','seasonal':'no','anomaly':'no','missing_block':'no'},
            'prediction':uniform_prediction() if pred is None else pred,
            'failure':{'type':'ToolError','reason':'test_failure'} if failed else None,
            'trace':[],'input_tokens':0,'output_tokens':0,'model_calls':[],'elapsed_ms':0}

class GenerationTests(unittest.TestCase):
    def test_reproducible(self):
        a=generate_case(1,'test','curved_chirp',19,96)
        b=generate_case(1,'test','curved_chirp',19,96)
        self.assertEqual(a,b)
    def test_independent_split(self):
        a=generate_case(1,'development','linear_sine',0,96)
        b=generate_case(1,'calibration','linear_sine',0,96)
        self.assertNotEqual(a['id'],b['id']);self.assertNotEqual(a['observation'],b['observation'])
    def test_heldout_families(self):
        cfg=config();validate_config(cfg)
        self.assertFalse(set(cfg['splits']['test'])&set(cfg['splits']['calibration']))
    def test_reject_overlap(self):
        cfg=config();cfg['splits']['test']=['linear_sine']
        with self.assertRaises(ValueError):validate_config(cfg)
    def test_reject_duplicate_family(self):
        cfg=config();cfg['splits']['test']*=2
        with self.assertRaises(ValueError):validate_config(cfg)
    def test_reject_bad_backend(self):
        cfg=config();cfg['agent']['backend']='pretend'
        with self.assertRaises(ValueError):validate_config(cfg)
    def test_all_factorial_controls(self):
        cases=[generate_case(5,'test','linear_sine',i,96) for i in range(36)]
        combos={(c['labels']['trend'],c['labels']['seasonal'],bool(c['metadata']['injected_indices']),c['metadata']['missing_mode']) for c in cases}
        self.assertEqual(len(combos),36)
    def test_public_data_excludes_truth(self):
        case=generate_case(1,'test','curved_chirp',0,96)
        self.assertEqual(set(public_record(case)),{'id','values'})
        self.assertEqual(set(vars(case['observation'])),{'values'})
    def test_missing_and_anomaly_labels(self):
        for i in range(36):
            c=generate_case(4,'test','amplitude_ar',i,96)
            self.assertEqual(c['labels']['anomaly']=='yes',bool(c['metadata']['observed_injections']))
            self.assertEqual(c['labels']['missing_block']=='yes',longest_run(v is None for v in c['observation'].values)>=8)
    def test_all_families_finite(self):
        cfg=config()
        for split in cfg['splits']:
            for c in generate_split(cfg,split):json.dumps(public_record(c),allow_nan=False)
    def test_invalid_observation(self):
        for values in ([float('nan')]*64,[True]*64,[0]*4):
            with self.assertRaises(ValueError):Observation(tuple(values))

class ToolTests(unittest.TestCase):
    def test_perfect_trend(self):
        a=np.linspace(0,5,96);ok=np.ones(96,dtype=bool)
        for robust in (False,True):self.assertAlmostEqual(trend_stats(a,ok,robust)['net_change'],5)
    def test_known_period(self):
        a=np.sin(2*np.pi*np.arange(128)/16)
        result=spectrum_stats(a)
        self.assertAlmostEqual(result['period'],16);self.assertGreater(result['peak_share'],.9)
    def test_missing_not_zero(self):
        values=[1.]*64;values[20:30]=[None]*10
        a,ok=filled(Observation(tuple(values)))
        self.assertTrue(np.allclose(a,1));self.assertEqual(ok.sum(),54)
    def test_all_missing_failure(self):
        with self.assertRaises(ToolError):filled(Observation(tuple([None]*64)))
    def test_tool_allowlist(self):
        session=ToolSession(Observation(tuple([0.]*64)))
        for name in ('labels','open','eval','__dict__'):
            with self.assertRaises(ToolError):session.call(name)
    def test_budget(self):
        session=ToolSession(Observation(tuple([0.]*64)),1);session.call('quality')
        with self.assertRaises(ToolError):session.call('trend')
    def test_cache_cost_and_no_mutation(self):
        session=ToolSession(Observation(tuple([0.]*64)));r=session.call('quality');r['observed']=0
        self.assertEqual(session.call('quality')['observed'],64)
        self.assertEqual(sum(e['units'] for e in session.trace),1)
    def test_constant_finite(self):
        session=ToolSession(Observation(tuple([0.]*64)))
        for name in ('trend','spectrum','outliers','robust_trend','robust_outliers','spectral_stability'):
            json.dumps(session.call(name),allow_nan=False)

class AgentTests(unittest.TestCase):
    def test_verifier_can_retain_confident_disagreement(self):
        p=uniform_prediction();p['anomaly']={'no':.95,'yes':.05}
        evidence=uniform_prediction();evidence['anomaly']={'no':.02,'yes':.98}
        session=ToolSession(Observation(tuple([0.]*64)))
        with patch('tsbench.agents.from_evidence',return_value=evidence):q,details=verify(p,session)
        self.assertGreater(q['anomaly']['no'],q['anomaly']['yes'])
        self.assertIn('anomaly',details['disagreed_tasks']);self.assertNotIn('anomaly',details['changed_tasks'])
    def test_verifier_can_correct_weak_disagreement(self):
        p=uniform_prediction();p['anomaly']={'no':.6,'yes':.4}
        evidence=uniform_prediction();evidence['anomaly']={'no':.02,'yes':.98}
        session=ToolSession(Observation(tuple([0.]*64)))
        with patch('tsbench.agents.from_evidence',return_value=evidence):q,details=verify(p,session)
        self.assertEqual(q['anomaly'],evidence['anomaly']);self.assertIn('anomaly',details['changed_tasks'])
    def test_all_arms_valid(self):
        obs=generate_case(5,'development','linear_sine',20,96)['observation']
        for method in METHODS:
            result=run_method(obs,method,config()['agent'],42)
            self.assertIsNone(result['failure']);validate_prediction(result['prediction'])
    def test_same_proposal_reused(self):
        obs=generate_case(5,'test','curved_chirp',20,96)['observation'];cfg=config()['agent']
        base=run_method(obs,'agent',cfg,42);before=copy.deepcopy(base)
        verified=verified_from(base,obs,cfg)
        self.assertEqual(base,before);self.assertEqual(verified['verification']['before'],base['prediction'])
        self.assertEqual(verified['model_calls'],base['model_calls'])
    def test_failure_not_repaired_by_oracle(self):
        obs=Observation(tuple([None]*64));cfg=config()['agent']
        base=run_method(obs,'agent',cfg,42);verified=verified_from(base,obs,cfg)
        self.assertIsNotNone(verified['failure']);self.assertIsNone(verified['prediction'])
    def test_local_llm_endpoint_only(self):
        for endpoint in ('https://example.com/api/chat','http://localhost.evil/api/chat','http://user@localhost:11434/api/chat','http://localhost:11434/api/chat?x=1'):
            cfg=config()['agent'];cfg['endpoint']=endpoint
            with self.assertRaises(ValueError):OllamaController(cfg,42)
    def test_ollama_tool_loop_mocked(self):
        cfg=config()['agent'];cfg.update(backend='ollama',model='test-model')
        actions=[{'tool':'quality'},{'prediction':uniform_prediction()}]
        with patch.object(OllamaController,'request',side_effect=actions):
            r=run_method(Observation(tuple([0.]*64)),'agent',cfg,42)
        self.assertIsNone(r['failure']);self.assertEqual(r['trace'][0]['tool'],'quality')
    def test_ollama_invalid_output(self):
        cfg=config()['agent'];cfg.update(backend='ollama',model='test-model')
        with patch.object(OllamaController,'request',return_value={'code':'open(secret)'}):
            r=run_method(Observation(tuple([0.]*64)),'agent',cfg,42)
        self.assertIsNotNone(r['failure']);self.assertIsNone(r['prediction'])
    def test_ollama_step_budget(self):
        cfg=config()['agent'];cfg.update(backend='ollama',model='test-model',max_steps=1)
        with patch.object(OllamaController,'request',return_value={'tool':'quality'}):
            r=run_method(Observation(tuple([0.]*64)),'agent',cfg,42)
        self.assertEqual(r['failure']['reason'],'step_budget')
    def test_ollama_network_timeout_recorded(self):
        cfg=config()['agent'];cfg.update(backend='ollama',model='test-model')
        with patch.object(OllamaController,'request',side_effect=TimeoutError('timed out')):
            r=run_method(Observation(tuple([0.]*64)),'agent',cfg,42)
        self.assertEqual(r['failure']['type'],'TimeoutError')
    def test_unknown_method(self):
        with self.assertRaises(ValueError):run_method(Observation(tuple([0.]*64)),'bad',config()['agent'],42)

class MetricTests(unittest.TestCase):
    def test_invalid_probability(self):
        for value in (float('nan'),-1,2,True):
            p=uniform_prediction();p['trend']['up']=value
            with self.assertRaises(ValueError):validate_prediction(p)
    def test_probability_sum(self):
        p=uniform_prediction();p['trend']['up']=.9
        with self.assertRaises(ValueError):validate_prediction(p)
    def test_uniform_brier(self):
        scores=task_scores(record())
        self.assertAlmostEqual(scores[0]['brier'],2/3)
        self.assertAlmostEqual(scores[1]['brier'],.5)
    def test_perfect_predictions(self):
        r=record();r['prediction']={t:{c:float(c==r['labels'][t]) for c in cs} for t,cs in CLASSES.items()}
        s=summarize([r]);self.assertEqual(s['accuracy'],1);self.assertLess(s['brier'],1e-12)
    def test_failures_penalized(self):
        r=record(failed=True);r['prediction']=None;s=summarize([r])
        self.assertEqual(s['accuracy'],0);self.assertEqual(s['brier'],2);self.assertEqual(s['failure_rate'],1)
        self.assertIsNone(s['ece_successful_only'])
    def test_temperature_fitting_split_guard(self):
        with self.assertRaises(ValueError):fit_temperature([record(split='test')])
    def test_temperature_preserves_rank(self):
        p=uniform_prediction();p['trend']={'down':.1,'flat':.2,'up':.7}
        q=temperature_scale(p,3);self.assertEqual(max(q['trend'],key=q['trend'].get),'up')
        validate_prediction(q)
    def test_empty_calibration(self):
        self.assertEqual(fit_temperature([])['status'],'no_valid_calibration')
    def test_bootstrap_identical(self):
        records=[record(id=str(i)) for i in range(10)]
        s=paired_bootstrap(records,records,100)
        self.assertEqual(s['ci95'],[0,0])
    def test_bootstrap_reject_unpaired(self):
        with self.assertRaises(ValueError):paired_bootstrap([record(id='a')],[record(id='b')])
    def test_semantic_digest_ignores_latency(self):
        r=record();s=copy.deepcopy(r);s['elapsed_ms']=100
        self.assertEqual(digest(semantic_records([r])),digest(semantic_records([s])))

class EndToEndTests(unittest.TestCase):
    def test_smoke_artifacts_and_reproducibility(self):
        cfg=config();cfg['samples_per_family']=3;cfg['bootstrap_replicates']=20
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';run(cfg,a);run(cfg,b)
            ma=json.loads((a/'manifest.json').read_text());mb=json.loads((b/'manifest.json').read_text())
            self.assertEqual(ma['status'],'complete')
            self.assertEqual(ma['semantic_result_sha256'],mb['semantic_result_sha256'])
            self.assertEqual(ma['dataset_hashes'],mb['dataset_hashes'])
            self.assertEqual((a/'calibration.json').read_text(),(b/'calibration.json').read_text())
            self.assertTrue((a/'REPORT.md').is_file());self.assertTrue((a/'scores.csv').is_file())
            with self.assertRaises(FileExistsError):run(cfg,a)

if __name__=='__main__':unittest.main()
