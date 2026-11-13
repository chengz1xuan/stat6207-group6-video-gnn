"""Run: conda activate stat6207 && python -m streamlit run app.py"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'.project-deps'))
import pandas as pd
import streamlit as st
from model_service import ModelService
from experiment_results import available_experiments
from final_experiment import default_folder, report_matches, data_dictionary, showcase
from pipeline import CAT,TEXT

st.set_page_config(page_title='VideoGraph Lab · Audited v4',page_icon='◈',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1450px}h1{letter-spacing:-1px}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce2d9;border-radius:8px;padding:14px}
[data-testid="stMetricValue"]{font-size:1.8rem}div[data-testid="stTabs"] button{font-size:1rem}</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_service(folder, checkpoint_mtime):
    return ModelService(folder)

bundle_errors=[]
try:
    experiments=available_experiments(errors=bundle_errors)
except (OSError,ValueError,KeyError,TypeError) as error:
    st.error(f'主实验文件加载失败：{error}')
    st.stop()
labels=dict(experiments)
with st.sidebar:
    choices=list(labels)
    folder=st.selectbox('结果版本',choices,index=choices.index(default_folder(choices)),format_func=labels.get,key='experiment_folder')
if st.session_state.get('active_experiment')!=folder:
    for key in list(st.session_state):
        if key in {'last_prediction','graph_brand','graph_video','inspect_node',
                   'prediction_reference','graph_similarity'} or key.startswith(('edit_','duration_','age_','text_')):
            del st.session_state[key]
    st.session_state['active_experiment']=folder
try:
    service=load_service(folder,(Path(folder)/'primary_gnn.pt').stat().st_mtime_ns)
except (OSError,ValueError,KeyError,RuntimeError) as error:
    st.error(f'所选实验无法加载：{error}')
    st.stop()
author_brief=showcase(folder)
d=service.results;m=d['metrics'];selected=m['selected_primary_gnn'];r=m['test'][selected]
video_count=len(d['videos']);brand_count=len({v['brand_family'] for v in d['videos']})
split_label=f'{m["train_size"]} / {m["test_size"]}'
st.caption('STAT6207 · CHENG, Zixuan · HUANG, Ze · SONG, Mingkun · HOU, Kaiwen')
st.title('视频创意图实验室')
st.write('比较品牌、元数据和创意图信号，通过冻结 GNN 探索待检验的视频设置。')
for error in bundle_errors:
    st.warning(f'已排除未通过校验的附加实验（主实验仍可使用）：{error}')
if service.exploratory:
    st.warning(f'探索性修正实验：{service.manifest["split_counts"]["quarantine"]} 条待审视频隔离；'
               '样本、图策略与划分不同，不能与主实验直接作配对提升比较。')
elif len(experiments)==1:
    st.info('当前显示最新主分支的正式结果。未发现完整的附加探索性实验包；新增代码本身不包含明坤本机的实验结果。')
with st.sidebar:
    st.header('实验设置')
    st.write('**本地 Streamlit + Python**')
    st.caption('直接加载本地 PyTorch Geometric 模型与冻结实验数据。')
    st.write('模型：',service.name)
    st.write('目标：ln(1 + views)')
    st.write(f'分组划分：{m["train_size"]} 训练 / {m["test_size"]} 测试')
    st.caption('3 折调参只使用训练数据；最终在完整训练集重训。')
    st.divider()
    st.write('数据采集：2026-11-09')
    st.caption(f'当前纳入 {video_count} 条视频、{brand_count} 个品牌。评论计数缺失不代表零。')
    if service.exploratory:
        st.download_button('下载修正与待审记录',json.dumps(d['audit'],ensure_ascii=False,indent=2),
                           file_name='revision_audit.json',mime='application/json')
    if report_matches(folder):
        st.download_button('下载四页 IEEE 报告',(ROOT/'output/pdf/Group6_Final_Report.pdf').read_bytes(),
                           file_name='Group6_Final_Report.pdf',mime='application/pdf')
        st.caption('报告对应当前 v4 队列，并包含同队列双策略配对比较。')
    else:
        st.caption('当前为历史/其他实验；新版报告对应 v4，不混用此版本的结果。')
    with st.expander('作者与联系方式'):
        for author in json.loads((ROOT/'AUTHORS.json').read_text()):
            st.write(author['name']);st.caption(author['email'])

cols=st.columns(4)
for col,label,value in zip(cols,['真实播放量标签','品牌','分组训练 / 测试','主 GNN 测试 Log RMSE'],
                           [str(video_count),str(brand_count),split_label,f'{r["log_RMSE"]:.3f}']):
    col.metric(label,value)
results_tab,graph_tab,predict_tab,cases_tab,data_tab=st.tabs(['研究结果','图结构浏览','预测与优化建议','案例与 Storyboard','数据与报告'])

with results_tab:
    st.subheader('基线与图模型：同一测试集上的比较')
    st.caption('误差越低越好。主模型由训练集交叉验证选择，测试集不参与选择。')
    frame=pd.DataFrame(m['test']).T.rename_axis('Model').reset_index()
    st.dataframe(frame[['Model','log_MAE','log_RMSE','Spearman','views_MAE']],hide_index=True,width='stretch')
    st.bar_chart(frame.set_index('Model')[['log_RMSE']],color='#186b60',horizontal=True)
    st.info(f'当前主子图模型 Log RMSE 为 {r["log_RMSE"]:.3f}，视频节点结构模型为 {m["test"]["Node structural"]["log_RMSE"]:.3f}。主模型由训练集交叉验证选择。')
    if 'Brand mean' in m['test'] and 'Node no-brand edges' in m['test']:
        st.write(f'品牌均值基线 Log RMSE：{m["test"]["Brand mean"]["log_RMSE"]:.3f}；去同品牌连接的视频节点模型：{m["test"]["Node no-brand edges"]["log_RMSE"]:.3f}。')
    if 'brand_edge_audit' in d:
        edge_audit=d['brand_edge_audit']
        st.warning(f'视频节点原始结构图 {edge_audit["same_brand_fraction"]:.0%} 的连接为同品牌视频连接，去品牌边后剩 {edge_audit["no_brand_pairs"]} 对。它的成绩不能证明创意概念关系有效；去边模型仍保留品牌输入特征。')
    if not service.exploratory:
        st.caption('保诚 170 周年中英文活动已人工合并；这是检查过旧测试误差后的修正版内部评估，不是全新未接触的外部测试集。')
    ci=d['uncertainty']['test_log_RMSE_95'][selected]
    st.write(f'主模型 campaign-bootstrap 95% 区间：**[{ci[0]:.3f}, {ci[1]:.3f}]**。区间以当前模型、采集和测试划分为条件。')
    st.subheader('跨品牌压力测试 · Leave-one-brand-out')
    if d.get('lobo') is not None:
        st.dataframe(pd.DataFrame(d['lobo']).T.rename_axis('Model'),width='stretch')
        st.caption(f'每次完全留出一个品牌；其余品牌内部选择停止轮次。这里报告 {brand_count} 个品牌的宏平均，区别于主分组划分结果。')
    else:
        st.warning('本实验未提供跨品牌压力测试结果。')
    st.subheader('误差切片与潜在改进')
    st.dataframe(pd.DataFrame(d['slices'])[['slice','n','log_MAE','log_RMSE']],hide_index=True,width='stretch')
    if not service.exploratory:
        audit=json.loads((ROOT/'artifacts/v2/reupload_error_audit.json').read_text(encoding='utf-8'))
        st.write(f'修正前的最大误差案例：{audit["title"]}；频道 {audit["channel_name"]}，实际 {audit["observed_views"]:,} 次，原预测 {audit["pre_correction_predicted_views"]:,.0f} 次。第三方转载的分发和曝光条件与官方广告不同；1996 是创意年份，不是本次上传日期。')
        st.caption('此数值是修正前的历史诊断，当前修正划分的预测另见结果 CSV。频道或曝光差异是潜在解释，未识别其因果作用。')
    elif d.get('largest_error'):
        error=d['largest_error']
        st.write(f'当前最大测试 log 误差：{error["title"]}；实测 {error["views"]:,.0f}，预测 {error["predicted_views"]:,.0f}；频道 {error["channel_name"]}。频道和分发差异是潜在解释，未识别其因果作用。')
    st.write('媒体投放、频道受众、曝光时间、人物吸引力及抽取噪声都可能影响播放量。应补充固定观察窗口、媒体预算和人工核验的 campaign 分组，并扩展独立外部测试集。')

with graph_tab:
    st.subheader('实际用于 GNN 的视频子图')
    gcols=st.columns([1,3])
    brand=gcols[0].selectbox('品牌筛选',['全部']+sorted({v['brand_family'] for v in d['videos']}),key='graph_brand')
    video_ids=[v['video_id'] for v in d['videos'] if brand=='全部' or v['brand_family']==brand]
    vid=gcols[1].selectbox('选择视频',video_ids,format_func=lambda k:f'[{service.videos[k]["split"]}] {service.videos[k]["title"]}',key='graph_video')
    row=service.videos[vid];graph=service.graphs[vid]
    sim=st.toggle('显示新增的语义相似边',value=True,key='graph_similarity')
    st.caption('该开关只改变图的显示。预测页始终使用已训练主模型所对应的边定义。')
    dot,stats=service.graph_dot(row,graph,sim)
    c1,c2,c3=st.columns(3)
    c1.metric('节点',stats['nodes']);c2.metric('原始结构关系',stats['source_relations']);c3.metric('新增无向相似边',stats['added_similarity_links'])
    st.graphviz_chart(dot,width='stretch',height=560)
    st.caption('深绿：视频根节点；浅黄：创意概念；浅绿：文本证据。橙色虚线表示新增启发式相似边。核心边保留原始 candidate_edges 关系与来源 ID。')
    left,right=st.columns([2,1])
    node_id=left.selectbox('检查节点',list(range(len(graph['nodes']))),format_func=lambda i:graph['nodes'][i]['field']+' · '+(graph['nodes'][i]['text'][:55] or row['title'][:55]),key='inspect_node')
    node=graph['nodes'][node_id];left.write(node['text'] or row['title']);left.caption(node['id'])
    linked=[e for e in graph['edges'] if node['id'] in [e['source'],e['target']]]
    left.dataframe(pd.DataFrame(linked),hide_index=True,width='stretch')
    right.metric('原始观测播放量',f'{int(row["views"]):,}')
    right.caption('采集时间：'+row['captured_at']);right.link_button('查看 YouTube 原视频',row['source_url'])

with predict_tab:
    st.subheader('修改创意设置，直接运行本地 GNN')
    options=['__new__']+list(service.videos)
    reference=st.selectbox('输入来源',options,format_func=lambda k:'新视频 Brief · 只使用声明的设置与文本' if k=='__new__' else f'[{service.videos[k]["split"]}] {service.videos[k]["title"]}',key='prediction_reference')
    template=(author_brief['row'] if author_brief else next(v for v in d['videos'] if v['split']=='train')) if reference=='__new__' else service.videos[reference]
    ages=service.age_support
    st.info(f'目标是采集时的累计播放量，年龄只是输入协变量。训练集曝光年龄范围 {ages["training_min_days"]:.1f}–{ages["training_max_days"]:.1f} 天，中位数 {ages["training_median_days"]:.1f} 天；没有验证上线后 30 天的固定窗口预测。')
    with st.form('prediction_settings'):
        edited=dict(template)
        fields=st.columns(4)
        for i,name in enumerate(CAT):
            categories=list(service.features.root.enc.categories_[i]);value=template[name]
            if value not in categories:categories.append(value)
            edited[name]=fields[i%4].selectbox(name,categories,index=categories.index(value),key=f'edit_{reference}_{name}')
        c1,c2=st.columns(2)
        edited['duration_seconds']=c1.number_input('时长（秒）',min_value=1.,max_value=3600.,value=30. if reference=='__new__' else max(1.,float(template['duration_seconds'] or 30)),key=f'duration_{reference}')
        edited['age_days']=c2.number_input('采集对应的曝光年龄（天；不是预测窗口）',min_value=0.,max_value=20000.,value=ages['training_median_days'] if reference=='__new__' else float(template['age_days'] or 0),key=f'age_{reference}')
        texts=st.columns(2)
        for i,name in enumerate(TEXT):edited[name]=texts[i%2].text_area(name,value=template.get(name,''),height=100,key=f'text_{reference}_{name}')
        c1,c2=st.columns(2)
        submitted=c1.form_submit_button('预测播放量',type='primary',width='stretch')
        optimized=c2.form_submit_button('搜索支持的创意设置',width='stretch')
    st.caption('现有视频的抽取证据保持固定，编辑后的分类和文案需由人核验一致性。新视频不借用其他视频的证据节点。')
    if submitted or optimized:
        if reference=='__new__':
            edited={k:edited[k] for k in CAT+TEXT+['duration_seconds','age_days']}
            edited['title']='New advertising brief';edited['video_id']='new-video-example'
        graph=service.query_graph(edited,None if reference=='__new__' else reference)
        result=service.optimize(edited,graph) if optimized else {'base':service.predict(edited,graph)}
        st.session_state['last_prediction']={'experiment':folder,'reference':reference,'input':edited,'result':result}
    saved=st.session_state.get('last_prediction')
    if saved and saved.get('experiment')==folder and saved['reference']==reference:
        pred=saved['result']['base'];st.metric('模型预测累计播放量',f'{pred["views"]:,.0f}')
        diag=pred['age_diagnostic']
        if diag['outside_training_range']:
            st.warning(f'范围外推：输入年龄 {diag["input_age_days"]:.1f} 天超出训练年龄范围。此结果只供探索，未经固定窗口或 30 天预测验证；OOF 残差范围也不保证适用于此输入。')
        else:
            st.caption('年龄位于训练范围内也不代表新 Brief 或该时间点已获得独立验证。')
        st.write(f'ln(1 + views) = {pred["log"]:.4f}')
        st.write(f'OOF 残差敏感度范围：**{pred["band"][0]:,.0f} – {pred["band"][1]:,.0f}**')
        st.caption('这是名义 90% 的训练折外绝对 log 残差范围，不保证个体覆盖率；log 预测的反变换也不是无偏的平均播放量。')
        if 'best' in saved['result']:
            outcome=saved['result'];best=outcome['best']
            st.metric('支持范围内的最佳预测',f'{best["views"]:,.0f}')
            if best['changes']:st.dataframe(pd.DataFrame([{'设置':k,'建议值':v} for k,v in best['changes'].items()]),hide_index=True)
            else:st.write('没有支持的候选设置提高预测，保留当前设置。')
            st.caption(f'评估了 {outcome["candidates"]} 个候选；最多改变 mechanic、structure、pacing 中的两项。候选三元组在训练集中至少出现两次。')
            st.write('**支持该组合的训练集来源**')
            for item in best['support_examples']:st.link_button(item['title'],item['source_url'])
        st.download_button('下载本次预测 JSON',json.dumps(saved,ensure_ascii=False,indent=2),file_name='video_prediction.json',mime='application/json')
    st.info('预测提升是关联与待检验的假设，没有观测到修改后视频的实际播放量，也不能据此宣称销售、品牌认知或因果提升。')

with cases_tab:
    st.subheader('新视频 Brief 与 30 秒 Storyboard')
    new=author_brief or d['new_video'];st.write('**目标：**',new['brief_objective']);st.write('**品牌 / 产品语境：**',new['brand'])
    c1,c2=st.columns(2);c1.metric('原设置预测',f'{new["base_prediction"]:,.0f}');c2.metric('建议设置预测',f'{new["optimized_prediction"]:,.0f}')
    if author_brief:st.caption(author_brief['input_provenance'])
    st.json(new['changes']);st.dataframe(pd.DataFrame(new['storyboard']),hide_index=True,width='stretch')
    case_age=new.get('age_days',30.)
    st.caption(f'此案例输入曝光年龄为 {case_age:.1f} 天，30 秒指视频时长。预测目标为累计播放量，并非经验证的固定窗口预测；新视频没有实测响应，声明图与训练抽取图也存在差异。')
    st.subheader(f'{len(d["cases"])} 个当前实验测试案例')
    for i,case in enumerate(d['cases'],1):
        with st.expander(f'{i}. {case["brand"]} · {case["title"]}',expanded=i==1):
            st.write('**Brief / objective：**',case['brief_objective'])
            st.write('**原始创意元素：**');st.json(case['creative_elements'])
            cols=st.columns(3)
            for c,label,value in zip(cols,['原视频实测','原设置预测','建议设置预测'],[case['observed_views'],case['base_prediction'],case['optimized_prediction']]):c.metric(label,f'{value:,.0f}')
            st.caption(f'原预测残差范围：{case["uncertainty_band"][0]:,.0f} – {case["uncertainty_band"][1]:,.0f}，不保证个体覆盖率。')
            st.json(case['changes']);st.write('**可迁移启示：**',case['transferable_takeaway'])
            for item in case['support_examples']:st.link_button(item['title'],item['source_url'])

with data_tab:
    st.subheader('冻结数据、采集证据与下载')
    st.write('一次采集覆盖 219 个链接；214 个链接取得精确播放量，其余 5 个没有混用历史计数。')
    snapshot=pd.read_csv(ROOT/'data/youtube/snapshot.csv')
    st.dataframe(snapshot[['youtube_id','views','likes','comments','channel_name','retrieved_at_utc','status']],hide_index=True,width='stretch')
    evidence_id=st.selectbox('查看一条采集证据',snapshot.youtube_id.tolist(),key='source_evidence')
    st.json(json.loads((ROOT/'data/youtube'/f'{evidence_id}.json').read_text(encoding='utf-8')))
    c1,c2,c3=st.columns(3)
    c1.download_button('下载冻结指标 CSV',(ROOT/'data/youtube/snapshot.csv').read_bytes(),file_name='youtube_snapshot.csv',mime='text/csv')
    dictionary=data_dictionary(service.folder)
    c2.download_button('下载当前实验数据字典',dictionary.read_text(encoding='utf-8'),file_name='DATA_DICTIONARY.md',mime='text/markdown')
    c3.download_button('下载当前实验指标 JSON',(service.folder/'metrics.json').read_bytes(),file_name='metrics.json',mime='application/json')
    with st.expander('当前实验与数据字典'):st.markdown(dictionary.read_text(encoding='utf-8'))
    st.caption('报告下载与对应实验绑定；所有预测均由本机 Python 模型计算。')
