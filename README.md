# DocVisRAG

褰撳墠闃舵锛?*闃舵 6 - 绔埌绔妯℃€?RAG 闂瓟**

## 椤圭洰绠€浠?DocVisRAG 鏄竴涓潰鍚戝鏉?PDF銆佹壂鎻忎欢銆丳PT 鎴浘绛夋枃妗ｇ殑澶氭ā鎬佹枃妗ｉ棶绛旂郴缁熴€? 
褰撳墠闃舵瀹炵幇鈥滈〉闈㈡憳瑕?+ OCR 鏂囨湰鈥濈殑椤甸潰绾ф绱㈠熀绾匡紝涓嶇敓鎴愭渶缁堢瓟妗堛€?
## Docker 杩愯鏂瑰紡
```bash
docker run --gpus all --ipc=host --network=host -it --rm \
  -v /data3/zengjian/DocVisRAG:/workspace/DocVisRAG \
  -v /data3/zengjian/.cache:/root/.cache \
  -w /workspace/DocVisRAG \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_ENDPOINT=https://hf-mirror.com \
  docvisrag:cu124
```

## 闃舵 0锛氱幆澧冩鏌?```bash
python scripts/env/check_env.py
```

## 闃舵 1锛氬崟鍥?VLM 闂瓟
```bash
python scripts/qa/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "璇锋鎷繖寮犲浘鐗囩殑涓昏鍐呭銆?
```

## 闃舵 2锛氭枃妗ｆ憚鍏ヤ笌椤甸潰娓叉煋
```bash
python scripts/ingest/ingest_render.py \
  --input data/samples/demo.pdf \
  --output data/outputs/demo_pages \
  --dpi 180
```

## 闃舵 3锛氶〉闈㈢骇 VLM 闂瓟
```bash
python scripts/qa/page_qa.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --page 1 \
  --question "杩欎竴椤典富瑕佽浜嗕粈涔堬紵"
```

## 闃舵 4锛歄CR 涓庣函鏂囨湰妫€绱㈠熀绾?```bash
python scripts/ingest/run_ocr.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/ocr.jsonl

python scripts/retrieve/build_text_index.py \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --index-dir data/indexes/demo_text

python scripts/retrieve/text_search.py \
  --index-dir data/indexes/demo_text \
  --question "鏈枃妗ｇ殑涓昏缁撹鏄粈涔堬紵" \
  --top-k 5
```

## 闃舵 5锛氳交閲忓妯℃€侀〉闈㈢储寮?
### 1) 鐢熸垚姣忛〉 VLM 鎽樿
```bash
python scripts/ingest/build_page_summaries.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/page_summaries.jsonl
```

杈撳嚭 `page_summaries.jsonl`锛屾瘡琛屽寘鍚細
- `doc_id`
- `page_index`
- `image_path`
- `summary`

### 2) 寤虹珛椤甸潰绾ф贩鍚堢储寮曪紙鎽樿 + OCR锛?```bash
python scripts/retrieve/build_hybrid_index.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --summaries data/outputs/demo_pages/page_summaries.jsonl \
  --index-dir data/indexes/demo_hybrid
```

绱㈠紩鐩綍杈撳嚭锛?- `index.faiss`
- `metadata.jsonl`
- `config.json`

### 3) 椤甸潰绾ф绱?```bash
python scripts/retrieve/hybrid_search.py \
  --index-dir data/indexes/demo_hybrid \
  --question "鍥捐〃灞曠ず浜嗕粈涔堣秼鍔匡紵" \
  --top-k 3
```

杩斿洖瀛楁鍖呭惈锛?- `page_index`
- `image_path`
- `summary`
- `ocr_text_preview`
- `score`

## 闃舵 6锛氱鍒扮澶氭ā鎬?RAG 闂瓟

杈撳叆鐢ㄦ埛闂鍚庯紝绯荤粺浼氳嚜鍔細
1. 浠?hybrid index 妫€绱?top-k 椤甸潰锛?2. 璇诲彇鍊欓€夐〉闈㈠浘鍍忋€侀〉闈㈡憳瑕併€丱CR 鏂囨湰锛?3. 浜ょ粰 VLM 鐢熸垚甯﹀紩鐢ㄩ〉鐮佺殑绛旀銆?
```bash
python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --question "绗?2 鑺傜殑涓昏缁撹鏄粈涔堬紵" \
  --top-k 3
```

鍙€夊弬鏁帮細
- `--model-id`
- `--load-in-4bit`
- `--max-new-tokens`

杈撳嚭鍖呭惈锛?- `绛旀锛歚
- `渚濇嵁锛歚
- `寮曠敤锛歚
- `涓嶇‘瀹氭€э細`

鍏朵腑寮曠敤鎸夆€滅 X 椤碘€濇牸寮忚緭鍑恒€?
## 褰撳墠闃舵涓嶅寘鍚?- ColPali 鎺ュ叆
- Web UI

## 娉ㄦ剰
- 涓嶈鎶婃ā鍨嬫潈閲嶆彁浜よ繘浠撳簱銆?
## 闃舵 7锛欸radio Demo

杩愯锛?
```bash
python app.py
```

鍚姩鍚庢墦寮€锛?
```text
http://localhost:7860
```

Demo 鍔熻兘锛?- 涓婁紶 PDF/PNG/JPG/JPEG/WEBP銆?- 鐐瑰嚮鈥滄瀯寤虹储寮曗€濊嚜鍔ㄦ墽琛岋細鏂囨。娓叉煋 -> OCR -> 椤甸潰鎽樿 -> Hybrid 绱㈠紩銆?- 杈撳叆闂骞剁偣鍑烩€滄彁闂€濓紝杩斿洖绛旀銆佷緷鎹€佸紩鐢ㄩ〉鐮佸拰涓嶇‘瀹氭€с€?- 椤甸潰杩囧鏃讹紝Demo 榛樿鍙鐞嗗墠 10 椤靛苟鍦ㄦ棩蹇楁彁绀恒€?

## 闃舵 8锛氬疄楠岃瘎娴嬭剼鏈?
### 1) 鍑嗗灏忚妯℃祴璇曢泦锛堝缓璁厛鍋?20 鏉★級

鍦?`eval/questions.example.jsonl` 鍩虹涓婃墿灞曚负浣犺嚜宸辩殑 `eval/questions.jsonl`锛屾瘡琛屼竴涓牱鏈細

```json
{
  "id": "q001",
  "doc_path": "data/samples/demo.pdf",
  "question": "杩欎唤鏂囨。鐨勪富瑕佺粨璁烘槸浠€涔堬紵",
  "answer": "鏍囧噯绛旀",
  "evidence_pages": [1, 2],
  "type": "summary"
}
```

瀛楁寤鸿锛?- `id`: 鍞竴闂缂栧彿銆?- `doc_path`: 瀵瑰簲鏂囨。璺緞銆?- `question`: 鐢ㄦ埛闂銆?- `answer`: 鏍囧噯绛旀锛堝敖閲忕畝娲併€佸彲鍒ゅ畾锛夈€?- `evidence_pages`: 鏀拺绛旀鐨勯〉鐮侊紙1-based锛夈€?- `type`: `text/table/chart/summary/layout` 涔嬩竴銆?
20 鏉℃祴璇曢泦寤鸿閰嶆瘮锛?- `text`: 6 鏉?- `summary`: 4 鏉?- `table`: 4 鏉?- `chart`: 4 鏉?- `layout`: 2 鏉?
### 2) 妫€绱㈣瘎娴嬶紙Recall@K + MRR锛?
```bash
python scripts/eval/eval_retrieval.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --out data/outputs/eval_retrieval.json
```

杈撳嚭锛?- 鎬讳綋锛歚Recall@1/3/5`銆乣MRR`
- 鍒嗙被鍨嬬粺璁★紙鎸?`type`锛?- 姣忎釜闂鐨勬绱㈡槑缁?
### 3) 闂瓟璇勬祴锛圗M/F1/ANLS锛?
```bash
python scripts/eval/eval_qa.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --out data/outputs/predictions.jsonl \
  --limit 10
```

杈撳嚭锛?- `predictions.jsonl`锛氭瘡棰橀娴嬬瓟妗堛€佸紩鐢ㄩ〉鐮併€佽瘉鎹〉鐮併€丒M/F1/ANLS
- `predictions.jsonl.summary.json`锛氭€讳綋鍧囧€肩粺璁?
### 4) 閿欒鍒嗘瀽鏂囨。

```bash
python scripts/eval/make_error_analysis.py \
  --predictions data/outputs/predictions.jsonl \
  --out data/outputs/error_analysis.md
```

閿欒鍒嗘瀽妯℃澘鍖呭惈锛?- OCR 璇嗗埆閿欒
- 妫€绱㈡湭鍙洖姝ｇ‘椤甸潰
- 妫€绱㈡帓搴忛敊璇?- 鍥捐〃/琛ㄦ牸璇诲彇閿欒
- 鐢熸垚妯″瀷骞昏
- 寮曠敤椤电爜閿欒
- 鏍囧噯绛旀鎴栨爣娉ㄤ笉娓呮

### 5) 鎸囨爣瑙ｉ噴

- `Recall@K`锛氬墠 K 涓绱㈤〉鏄惁鍛戒腑鑷冲皯涓€涓爣娉ㄨ瘉鎹〉锛堣秺楂樿秺濂斤級銆?- `MRR`锛氱涓€涓懡涓〉鐨勫€掓暟鎺掑悕鍧囧€硷紙瓒婇珮瓒婂ソ锛夈€?- `EM`锛氶娴嬬瓟妗堜笌鏍囧噯绛旀瑙勮寖鍖栧悗瀹屽叏涓€鑷存瘮渚嬨€?- `F1`锛氶娴嬩笌鏍囧噯绛旀鐨?token 閲嶅彔骞宠　鎸囨爣銆?- `ANLS`锛氬熀浜庣紪杈戣窛绂荤殑杩戜技鍖归厤鍒嗘暟锛堜綆浜庨槇鍊艰 0锛夈€?
寤鸿鍦ㄦ姤鍛婁腑鍚屾椂缁欏嚭锛?- 鎬讳綋鎸囨爣
- 鍒嗙被鍨嬫寚鏍囷紙text/table/chart/summary/layout锛?- 鍏稿瀷閿欒妗堜緥涓庢敼杩涙柟鍚?

### 6) 涓€閿嚜鍔ㄥ寲璇勬祴锛堟帹鑽愶級

鏂板鑴氭湰锛歚scripts/bench/run_benchmark_suite.py`

浣滅敤锛氳嚜鍔ㄦ墽琛?- OCR
- 椤甸潰鎽樿
- Hybrid 绱㈠紩鏋勫缓
- 妫€绱㈣瘎娴?- 闂瓟璇勬祴
- 閿欒鍒嗘瀽

骞舵妸涓棿鏂囦欢鍜岀粨鏋滄枃浠舵暣鐞嗗湪鍚屼竴瀛愮洰褰曚笅锛岀洰褰曠粨鏋勫涓嬶細

```text
data/bench_runs/<suite_name>/<benchmark_name>/
  inputs/
    manifest.snapshot.json
    questions.jsonl
  intermediate/
    ocr.jsonl
    page_summaries.jsonl
    hybrid_index/
  results/
    eval_retrieval.json
    predictions.jsonl
    predictions.summary.json
  reports/
    error_analysis.md
  logs/
    pipeline.log
  run_meta.json
  summary.json
```

suite 鏍圭洰褰曡繕浼氱敓鎴愶細
- `overview.json`
- `<suite_name>.tar.gz`锛堣嚜鍔ㄦ墦鍖咃紝鏂逛究褰掓。锛?
鍗曟暟鎹泦杩愯绀轰緥锛?
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --out-root data/bench_runs \
  --qa-limit 50
```

澶氭暟鎹泦鎵归噺杩愯绀轰緥锛?
```bash
python scripts/bench/run_benchmark_suite.py \
  --batch-config eval/bench_suite.example.json \
  --out-root data/bench_runs \
  --qa-limit 50
```

鎵归噺閰嶇疆妯℃澘瑙侊細`eval/bench_suite.example.json`銆?

### 7) 缂烘枃浠惰嚜鍔ㄤ笅杞戒笌杞崲

`run_benchmark_suite.py` 鐜板湪鏀寔鍦ㄧ己灏?`manifest.json` / `questions.jsonl` 鏃惰嚜鍔ㄥ噯澶囨暟鎹€?
渚嬪鐩存帴璺?DocVQA锛?
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --out-root data/bench_runs
```

鑻ヤ笂杩拌緭鍏ユ枃浠朵笉瀛樺湪锛岃剼鏈細鑷姩璋冪敤 `scripts/bench/prepare_benchmark.py` 涓嬭浇骞惰浆鎹紝鍐嶇户缁?OCR/绱㈠紩/璇勬祴鍏ㄦ祦绋嬨€?
浠呬笅杞藉皯閲忔牱鏈敤浜庡啋鐑熸祴璇曪細

```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --auto-prepare-limit 50 \
  --out-root data/bench_runs
```

瀹屽叏鍏抽棴鑷姩鍑嗗锛堜弗鏍艰姹傝緭鍏ュ凡瀛樺湪锛夛細

```bash
python scripts/bench/run_benchmark_suite.py ... --no-auto-prepare-missing
```

### 8) 浜х墿缁勭粐涓庤嚜鍔ㄥ噯澶囦紭鍖?
宸蹭紭鍖栵細
- 榛樿涓嶅啀鐢熸垚鍘嬬缉鍖咃紱浠呭綋浼?`--archive` 鎵嶆墦鍖呫€?- `run_meta.json` / `summary.json` 澧炲姞 `benchmark`銆乣dataset_id`銆乣split` 瀛楁銆?- 鏀寔 `chartvqa` 浣滀负 `chartqa` 鐨勫埆鍚嶃€?- 鏂板 `--force-prepare`锛氬嵆浣垮凡鏈?`data/bench/...` 涔熷己鍒堕噸鏂颁笅杞藉苟杞崲銆?
璇存槑锛?- 濡傛灉涔嬪墠鐢?`--auto-prepare-limit 3` 璺戣繃锛宍data/bench/docvqa` 浼氫繚鎸佸皬鏍锋湰锛屽悗缁笉鍔?`--force-prepare` 浼氱户缁鐢ㄨ灏忔牱鏈€?- 鎯宠窇鍏ㄩ噺锛岃鍘绘帀 limit 骞跺姞 `--force-prepare`銆?
绀轰緥锛堝叏閲?DocVQA锛岄粯璁や笉鎵撳寘锛夛細

```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --force-prepare \
  --out-root data/bench_runs
```

鑻ユ兂鎵撳寘鍐嶅姞锛?
```bash
--archive
```

## Benchmark Suite Output (Updated)

- By default, `scripts/bench/run_benchmark_suite.py` does **not** create a `.tar.gz` archive.
- If you need an archive, add `--archive` explicitly.
- Each suite now includes:
  - `benchmarks.md` (run_name / benchmark / dataset_id / split / qa_success)
  - `README.md` (directory layout and notes)
- Each benchmark run now includes:
  - `dataset_format.md` (manifest/questions/results schema)

### Auto-prepare Path Rules

- Single-run auto prepare now writes to `data/bench/<name>/...` based on the `--name` you pass.
- Example: `--name chartvqa` writes to `data/bench/chartvqa/`.
- Internally, `chartvqa` is still mapped to the ChartQA dataset for loading.

### Why the prepared benchmark looked small

- If `--auto-prepare-limit` was used before, existing prepared files may stay small.
- Use `--force-prepare` to rebuild benchmark files from scratch.


## Stage 9 (Optional): Visual Retriever (ColPali/Byaldi style)

This is an optional enhancement. Existing hybrid index remains the default fallback.
If visual dependencies are not installed, stage 6/7/8 hybrid flow still works.

### Retriever types
- `hybrid`: page summary + OCR text (existing default)
- `visual`: page-level visual retriever
- `fusion`: reciprocal-rank-fusion of hybrid + visual

### Build visual index
```bash
python scripts/retrieve/build_visual_index.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --index-dir data/indexes/demo_visual
```

### Visual search
```bash
python scripts/retrieve/visual_search.py \
  --index-dir data/indexes/demo_visual \
  --question "图表展示了什么趋势？" \
  --top-k 3
```

### DocQA switch retriever type
```bash
python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --visual-index-dir data/indexes/demo_visual \
  --retriever-type fusion \
  --question "第 2 节的主要结论是什么？" \
  --top-k 3
```

### Gradio demo switch
In `app.py`, retriever dropdown supports `hybrid / visual / fusion`.

### Benchmark switch
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type fusion
```
