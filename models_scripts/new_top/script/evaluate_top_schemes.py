#!/usr/bin/env python3
"""第四步：评测纯Embedding、旧融合和精确规则内部重排。"""
from __future__ import annotations
import argparse, csv, itertools, math
from collections import Counter, defaultdict
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent; NEW_TOP_DIR=SCRIPT_DIR.parent
DEFAULT_EMBEDDING=NEW_TOP_DIR/'output'/'embedding_top10.csv'; DEFAULT_RULES=NEW_TOP_DIR/'output'/'rule_matches.csv'; DEFAULT_FINAL=NEW_TOP_DIR/'output'/'final_top10.csv'; DEFAULT_LABELS=NEW_TOP_DIR/'evaluation'/'llm_blind_labels.csv'
STRATA=('A_no_injection','B_low_injection_rank1_kept','C_low_injection_rank1_dropped','D_heavy_injection_rank1_dropped','E_heavy_injection_rank1_kept')
def classify_final_group(rows):
    injected=sum(row['fusion_source']=='rule_only' for row in rows); dropped=not any(row['semantic_rank']=='1' for row in rows)
    if injected==0:return 'A_no_injection'
    if injected<=4 and not dropped:return 'B_low_injection_rank1_kept'
    if injected<=4:return 'C_low_injection_rank1_dropped'
    if dropped:return 'D_heavy_injection_rank1_dropped'
    return 'E_heavy_injection_rank1_kept'
def population_counts(path):
    counts=Counter()
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for _,group in itertools.groupby(csv.DictReader(f),key=lambda r:r['job_id']): counts[classify_final_group(list(group))]+=1
    return counts
def load_labels(path):
    required={'sample_id','stratum','job_id','major_code','llm_relevance_label','semantic_rank','final_rank'}
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); missing=required-set(reader.fieldnames or [])
        if missing: raise ValueError(f'盲审标签缺少字段：{sorted(missing)}')
        rows=list(reader)
    jobs,pairs=defaultdict(list),set()
    for row in rows:
        pair=(row['job_id'],row['major_code'])
        if pair in pairs: raise ValueError(f'盲审标签存在重复岗位—专业：{pair}')
        pairs.add(pair)
        if row['stratum'] not in STRATA or row['llm_relevance_label'] not in {'0','1','2'}: raise ValueError(f'盲审标签值无效：{pair}')
        jobs[row['job_id']].append(row)
    if len(jobs)!=50 or len(rows)!=704: raise ValueError(f'盲审规模异常：jobs={len(jobs)} rows={len(rows)}')
    return dict(jobs)
def load_output_rankings(path,job_ids):
    result=defaultdict(list); ranks=defaultdict(list)
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['job_id'] in job_ids: result[row['job_id']].append(row['major_code']); ranks[row['job_id']].append(int(row['final_rank']))
    for job_id in job_ids:
        if len(result[job_id])!=10 or sorted(ranks[job_id])!=list(range(1,11)): raise ValueError(f'最终结果岗位{job_id}不是完整Top10')
    return dict(result)
def dcg(labels): return sum((2**label-1)/math.log2(i+2) for i,label in enumerate(labels))
def per_job_metric(rows,ranking,k,threshold):
    labels={r['major_code']:int(r['llm_relevance_label']) for r in rows}; selected=[labels.get(c,0) for c in ranking[:k]]; relevant=[x for x in labels.values() if x>=threshold]; hits=sum(x>=threshold for x in selected); ideal=sorted(labels.values(),reverse=True)[:k]
    return {'precision':hits/k,'recall':hits/len(relevant) if relevant else 1.0,'ndcg':dcg(selected)/dcg(ideal) if dcg(ideal) else 1.0}
def weighted_metrics(jobs,rankings,population,k,threshold):
    by=defaultdict(list)
    for job_id,rows in jobs.items(): by[rows[0]['stratum']].append(per_job_metric(rows,rankings[job_id],k,threshold))
    total=sum(population.values())
    return {m:sum(population.get(s,0)/total*sum(v[m] for v in vals)/len(vals) for s,vals in by.items()) for m in ('precision','recall','ndcg')}
def label_rankings(jobs,field): return {jid:[r['major_code'] for r in sorted((r for r in rows if r[field]),key=lambda r:int(r[field]))] for jid,rows in jobs.items()}
def exact_targets(path):
    targets=defaultdict(set)
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['rule_match_type']=='1': targets[row['job_id']].add(row['major_code'])
    return dict(targets)
def exact_recall(path,targets,rank_field):
    hits,full=Counter(),Counter()
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for jid,group in itertools.groupby(csv.DictReader(f),key=lambda r:r['job_id']):
            if jid not in targets: continue
            rows=list(group)
            for k in (5,10):
                found={r['major_code'] for r in rows if int(r[rank_field])<=k}; count=len(found&targets[jid]); hits[k]+=count; full[k]+=int(count==len(targets[jid]))
    return {k:{'micro_recall':hits[k]/sum(map(len,targets.values())),'full_job_recall':full[k]/len(targets)} for k in (5,10)}
def main():
    parser=argparse.ArgumentParser(description='评测Top5/Top10及新旧融合策略'); parser.add_argument('--embedding',default=str(DEFAULT_EMBEDDING)); parser.add_argument('--rules',default=str(DEFAULT_RULES)); parser.add_argument('--final',default=str(DEFAULT_FINAL)); parser.add_argument('--labels',default=str(DEFAULT_LABELS)); parser.add_argument('--population-final',help='旧版融合结果，用于沿用原分层权重；不提供时按盲审样本分层权重'); args=parser.parse_args()
    embedding,rules,final,labels=map(lambda x:Path(x).resolve(),(args.embedding,args.rules,args.final,args.labels)); jobs=load_labels(labels); new_ranks=load_output_rankings(final,set(jobs))
    if args.population_final: population=population_counts(Path(args.population_final).resolve()); source='legacy_full_population'
    else: population=Counter(rows[0]['stratum'] for rows in jobs.values()); source='blind_sample'
    rankings={'pure_embedding':label_rankings(jobs,'semantic_rank'),'legacy_current_fusion':label_rankings(jobs,'final_rank'),'exact_rule_reorder':new_ranks}
    print(f'[population] source={source} counts={dict(population)}'); print(f'[blind_eval] jobs={len(jobs)} candidate_pairs={sum(map(len,jobs.values()))}')
    for name,threshold in (('broad_relevance',1),('core_relevance',2)):
        for scheme,ranking in rankings.items():
            for k in (5,10):
                m=weighted_metrics(jobs,ranking,population,k,threshold); print(f'[{name}] scheme={scheme} k={k} precision={m["precision"]:.6f} recall={m["recall"]:.6f} ndcg={m["ndcg"]:.6f}')
    targets=exact_targets(rules); print(f'[exact_benchmark] jobs={len(targets)} target_pairs={sum(map(len,targets.values()))}')
    for scheme,path,rank in (('pure_embedding',embedding,'semantic_rank'),('exact_rule_reorder',final,'final_rank')):
        for k,m in exact_recall(path,targets,rank).items(): print(f'[exact_recall] scheme={scheme} k={k} micro={m["micro_recall"]:.6f} full_job={m["full_job_recall"]:.6f}')
if __name__=='__main__': main()
