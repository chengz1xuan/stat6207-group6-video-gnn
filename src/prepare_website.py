from pathlib import Path
import shutil
r=Path(__file__).resolve().parents[1]
for src,dst in [('artifacts/v2/web_model.json','model.json'),('artifacts/v2/web_results.json','results.json'),('output/pdf/Group6_Final_Report.pdf','report.pdf'),('data/processed/DATA_DICTIONARY.md','data_dictionary.md')]:shutil.copy2(r/src,r/'website/dist'/dst)
print('Website synchronized to final v2 model, results, dictionary and IEEE report.')
