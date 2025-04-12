# ner-bert-crf
NER Model with BERT and CRF implemented with Pytorch

## architecture

Bert -> CRF

## dataset

NER标注工具: pip install label-studio，直接把train.txt导入

标注结果集train_label.json：

```
{
		"id": 4,
		"completed_by": 1,
		"result": [{
			"value": {
				"start": 2,
				"end": 4,
				"text": "北京",
				"labels": ["COUNTRY"]
			},
			"id": "bgqqtjGWxO",
			"from_name": "label",
			"to_name": "text",
			"type": "labels",
			"origin": "manual"
		}, {
			"value": {
				"start": 26,
				"end": 28,
				"text": "姚洋",
				"labels": ["PERSON"]
			},
			"id": "EEyq2k4ygn",
			"from_name": "label",
			"to_name": "text",
			"type": "labels",
			"origin": "manual"
		}],
		"was_cancelled": false,
		"ground_truth": false,
		"created_at": "2025-04-12T02:37:12.266779Z",
		"updated_at": "2025-04-12T02:37:12.266779Z",
		"draft_created_at": "2025-04-12T02:37:11.510776Z",
		"lead_time": 13.014,
		"prediction": {},
		"result_count": 2,
		"unique_id": "39024dba-7ab3-421b-b283-b032ca0720be",
		"import_id": null,
		"last_action": null,
		"bulk_created": false,
		"task": 1,
		"project": 3,
		"updated_by": 1,
		"parent_prediction": null,
		"parent_annotation": null,
		"last_created_by": null
	}],
	"file_upload": "cfd61273-train.txt",
	"drafts": [],
	"predictions": [],
	"data": {
		"text": "图为北京大学国家发展研究院教授、中国经济研究中心主任姚洋 图\/受访者供图"
	},
	"meta": {},
	"created_at": "2025-04-12T02:36:31.274078Z",
	"updated_at": "2025-04-12T02:37:12.291782Z",
	"inner_id": 1,
	"total_annotations": 1,
	"cancelled_annotations": 0,
	"total_predictions": 0,
	"comment_count": 0,
	"unresolved_comment_count": 0,
	"last_comment_updated_at": null,
	"project": 3,
	"updated_by": 1,
	"comment_authors": []
}
```

## training

```
python train.py
```

## inference

python test.py

```
s='特朗普拉起中美关税战，中国和欧洲会面意味着什么？美方会如何做出反应？'
result=ner(s)
print(result)
```

output:
```
[(0, 2, 'PERSON', '特朗普'), (11, 12, 'COUNTRY', '中国'), (14, 15, 'COUNTRY', '欧洲'), (24, 25, 'COUNTRY', '美方')]
```

## references

* [美团搜索中NER技术的探索与实践](https://tech.meituan.com/2020/07/23/ner-in-meituan-nlp.html)