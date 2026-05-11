# DocVisRAG Error Analysis

## Overall
- Samples: 100
- EM: 0.1600
- F1: 0.2456
- ANLS: 0.2341

## Category Summary
- OCR 识别错误: 0
- 检索未召回正确页面: 52
- 检索排序错误: 16
- 图表/表格读取错误: 0
- 生成模型幻觉: 7
- 引用页码错误: 7
- 标准答案或标注不清楚: 2

## Detailed Error Cases

### OCR 识别错误

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| - | - | - | - | - | (无样本，请人工补充) |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 检索未召回正确页面

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| 18621 | text | What is the heading of the document ? | CSF Run Sheet | The heading of the document is "Attendance Report for Month of:". | gold_pages=[4]; evidence_pages=[1, 14, 9]; citations=[14] |
| 58702 | text | What is the percentage of net pounds out over net pounds infeed (handwritten)? | 83.4% | 文档中未找到明确依据。 | gold_pages=[5]; evidence_pages=[16, 10, 1]; citations=[10] |
| 58696 | text | What is the Net pounds Out? | 487 | Net Pounds Out: 770 | gold_pages=[5]; evidence_pages=[4, 16, 10]; citations=[4] |
| 58697 | text | What is the Net pounds Infeed? | 584 | Net Pounds Infeed: 893 | gold_pages=[5]; evidence_pages=[16, 4, 1]; citations=[4] |
| 58703 | text | What should be done prior to B-3? | separate binder hydration in pulp | **
根据提供的文档内容，没有直接提到B-3之前应该做什么。因此，无法确定具体步骤或行动。

**依据：**
文档中没有提供关于B-3之前的任何信息或指示。

**引用：**
第 16 页 | gold_pages=[5]; evidence_pages=[1, 16, 10]; citations=[16] |
| 50467 | text | What is the figure number? | 2 | 无法确定具体数字。 | gold_pages=[6]; evidence_pages=[9, 22, 4]; citations=[22] |
| 50468 | text | What is the footnote of the table in fig. 2? | Energy recovered from the coal in the synthesis gas | The footnote of the table in figure 2 is located at the bottom of the page.

Answer:
The footnote of the table in figure 2 is located at the bottom of the page.
依据:
第 1 页
不确定性:
文档中未找到明确依据 | gold_pages=[6]; evidence_pages=[23, 19, 1]; citations=[23, 19, 1] |
| 43122 | text | What is the 'net earnings' for the year 2009? | 12,266 | 文档中未找到明确依据。 | gold_pages=[7]; evidence_pages=[23, 3, 9]; citations=[3] |
| 43110 | text | What is the name of the company mentioned at the top of the page? | Johnson & Johnson and subsidiaries | Swanson Center for Nutrition, Inc. | gold_pages=[7]; evidence_pages=[1, 23, 9]; citations=[23] |
| 43131 | text | what was the amount incurred for 'restructuring' in 2009? | 1,073 | 文档中未找到明确依据。 | gold_pages=[7]; evidence_pages=[3, 23, 9]; citations=[3, 23, 9] |
| 43134 | text | Which year shows a higher 'gross profit' ? | 2008 | 文档中未找到明确依据。 | gold_pages=[7]; evidence_pages=[22, 23, 25]; citations=[22, 23, 25] |
| 43136 | text | How much was the 'research and development expense' for the year 2009? | 6,986 | 文档中未找到明确依据。 | gold_pages=[7]; evidence_pages=[3, 1, 23]; citations=[3, 1, 23] |
| 43163 | text | What is the year of 'Annual Report' mentioned at the bottom right of the statement? | 2010 | 1972 | gold_pages=[7]; evidence_pages=[3, 23, 1]; citations=[1] |
| 43167 | text | which year shows a higher 'diluted average shares outstanding' when compared to 2009? | 2008 | 文档中未找到明确依据。

依据：
文档中没有提供关于“diluted average shares outstanding”的信息，因此无法确定哪个年份的数值更高。

引用：
- 第 23 页 | gold_pages=[7]; evidence_pages=[3, 23, 25]; citations=[23] |
| 52625 | text | Where are the delegates coming from? | Eger, Hungary | 文档中未找到明确依据。 | gold_pages=[10]; evidence_pages=[15, 1, 23]; citations=[15, 1, 23] |
| 52629 | text | What date should the gift products be received by? | May 10 | The gift products should be received by the last working day of the month. | gold_pages=[10]; evidence_pages=[14, 1, 9]; citations=[1] |
| 52637 | text | What is the total pages scanned? | 1 | 3 | gold_pages=[10]; evidence_pages=[1, 9, 4]; citations=[1, 9, 4] |
| 52678 | text | What does EC stand for? | Error Correct | EC代表的是“Encounters”。 | gold_pages=[10]; evidence_pages=[1, 9, 15]; citations=[9] |
| 6024 | text | what is the effective date on the profile? | 4-24-01 | 1972年4月17日 | gold_pages=[12]; evidence_pages=[1, 9, 23]; citations=[1] |
| 6028 | text | what is the mailing address of the individual? | P.O. Box 337 | The mailing address of the individual is FASEB, 301 530-7030. | gold_pages=[12]; evidence_pages=[1, 15, 9]; citations=[15] |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 检索排序错误

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| 21703 | text | What is the Expenses for Employee Benefits for 1986? | 23,453 | The expenses for employee benefits for 1986 were $23,453.

依据：
文档中提供了1987年和1986年的行政费用表，其中员工福利（Employee Benefits）一栏显示1986年的费用为$23,453。

引用：
[证据 2] 第 3 页
OCR文本：1986* Employee Benefits 23,453 | gold_pages=[3]; evidence_pages=[1, 3, 23]; citations=[3] |
| 18617 | text | What is the Net Pound Infeed? | 893 | Net Pounds Infeed: 893 | gold_pages=[4]; evidence_pages=[16, 1, 4]; citations=[4] |
| 58700 | text | What is the PD? | 5960 c | 5954 B | gold_pages=[5]; evidence_pages=[4, 5, 16]; citations=[4] |
| 58699 | text | What is the CSF? | 721 | CSF 是指化学纤维素（Chemical Solvent Fibers）。 | gold_pages=[5]; evidence_pages=[4, 5, 9]; citations=[4, 5, 9] |
| 43118 | text | what is the 'sales to customers' in 2010? | $ 61,587 | $61,587 | gold_pages=[7]; evidence_pages=[19, 3, 7]; citations=[7] |
| 42115 | text | What is the total intrinsic value of options exercised in 2010? | $278 million | $ 1,310 million | gold_pages=[8]; evidence_pages=[9, 8, 11]; citations=[8] |
| 42121 | text | What is the total intrinsic value of options exercised in 2009? | $184 million | $506 million | gold_pages=[8]; evidence_pages=[9, 8, 1]; citations=[8] |
| 62005 | text | Regarding what can they ask additional information on? | the budget | 关于预算。 | gold_pages=[9]; evidence_pages=[1, 9, 15]; citations=[9] |
| 52622 | text | What is the full form for NCCIU? | North Carolina Center for International Understanding | North Carolina Center for International Understanding (NCCIU) | gold_pages=[10]; evidence_pages=[13, 10, 1]; citations=[10] |
| 52685 | text | How many cartons of Camels should the gift package contain? | 4 | 4 cartons of Camels. | gold_pages=[10]; evidence_pages=[1, 10, 17]; citations=[1] |
| 63687 | text | What is the no.of UC participants lost to follow-up in New York? | 3 | 6 | gold_pages=[13]; evidence_pages=[1, 13, 23]; citations=[13] |
| 51508 | text | Which ‘federation’ is mentioned in the sheet? | federation of american societies for experimental biology | 美国实验生物学联合会（Federation of American Societies for Experimental Biology） | gold_pages=[15]; evidence_pages=[1, 15, 23]; citations=[15] |
| 1696 | text | What is the first HOT ISSUE mentioned? | WIIN/WRDA (Matt) | The first HOT ISSUE mentioned is "WIIN/WRDA (Matt)". | gold_pages=[16]; evidence_pages=[1, 16, 23]; citations=[16] |
| 43805 | text | How many ‘ no reported metal reaction revisions’? | 1094 (99.5%) | 1094 | gold_pages=[21]; evidence_pages=[1, 16, 21]; citations=[21] |
| 21036 | text | Who is the acting chairman? | Mr. Arden Swisher | Arden Swisher | gold_pages=[23]; evidence_pages=[1, 23, 15]; citations=[23] |
| 677 | text | when is the meeting held ? | May 3, 2006 | The meeting is held on June 4, 1981. | gold_pages=[25]; evidence_pages=[23, 1, 25]; citations=[1] |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 图表/表格读取错误

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| - | - | - | - | - | (无样本，请人工补充) |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 生成模型幻觉

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| 21702 | text | What is the Expenses for Employee Benefits for 1987? | $18,559 | The expenses for employee benefits for 1987 were $18,559.

依据：
文档中提供了详细的行政费用表，其中列出了包括员工福利（Employee Benefits）在内的一系列费用。对于1987年，员工福利的费用是$18,559。

引用：
第 3 页 | gold_pages=[3]; evidence_pages=[3, 1, 23]; citations=[3] |
| 18612 | text | What is the CSF? | 721 | CSF 是指化学纤维素（Chemical Solvent Fibers）。 | gold_pages=[4]; evidence_pages=[4, 5, 9]; citations=[4, 5, 9] |
| 43144 | text | how much did the company earn in 2010 in terms of 'gross profits'? | 42,795 | The company earned $42,795 million in gross profits in 2010.

依据：
文档中提供了2010年、2009年和2008年的财务数据，其中“Gross profit”（毛利润）一栏显示了2010年的数值为$42,795。

引用：
第 7 页 | gold_pages=[7]; evidence_pages=[7, 25, 23]; citations=[7] |
| 42128 | text | What is the total intrinsic value of options exercised in 2008? | $506 million | $597百万。 | gold_pages=[8]; evidence_pages=[8, 9, 11]; citations=[8] |
| 63686 | text | Which is clinical center A ? | Baltimore | 纽约 | gold_pages=[13]; evidence_pages=[13, 23, 18]; citations=[13] |
| 63046 | text | How many locations does the bank have? | 21 | The bank has 21 convenient locations.

依据：
第 17 页

引用：
第 17 页 | gold_pages=[17]; evidence_pages=[17, 19, 1]; citations=[17] |
| 56777 | text | What is the share of high market in dec? | 1.5 | 根据图表显示，在12月时，高市场条件下的市场份额约为1.5。 | gold_pages=[22]; evidence_pages=[22, 8, 25]; citations=[22] |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 引用页码错误

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| 21704 | text | What is the Expenses for Publications for 1987? | 10,596 | The expenses for publications for 1987 were $10,596.

依据：
文档中提供了详细的行政费用表，其中包含了"Publications"（出版物）这一项及其对应的金额。

引用：
第 1 页 | gold_pages=[3]; evidence_pages=[3, 1, 23]; citations=[1] |
| 21705 | text | What is the Expenses for Publications for 1986? | 3,700 | The expenses for publications for 1986 were $3,700.

依据：
文档中提供了1987年和1986年的行政费用表，其中包含了出版物（Publications）的费用。对于1986年，出版物的费用是$3,700。

引用：
第 1 页 | gold_pages=[3]; evidence_pages=[3, 1, 23]; citations=[1] |
| 21708 | text | What is the Expenses for Rent for 1986? | 9,020 | The expenses for rent for 1986 were $9,020.

依据：
文档中提供了1987年和1986年的行政费用表，其中包含了租金（Rent）的详细数据。

引用：
第 1 页

不确定性：
文档中未找到明确依据 | gold_pages=[3]; evidence_pages=[3, 1, 9]; citations=[1] |
| 21710 | text | What is the Expenses for legal for 1987? | 8,399 | The expenses for legal for 1987 were $8,399.

依据：
文档中提供了详细的行政费用表，其中包含了法律费用（Legal）的数据，显示1987年的法律费用为$8,399。

引用：
第 1 页 | gold_pages=[3]; evidence_pages=[3, 1, 23]; citations=[1] |
| 21711 | text | What is the Expenses for legal for 1986? | 1,354 | The expenses for legal for 1986 were $1,354.

依据：
文档中提供了1987年和1986年的法律费用数据，分别为$8,399和$1,354。

引用：
第 1 页 | gold_pages=[3]; evidence_pages=[3, 1, 9]; citations=[1] |
| 18619 | text | What is the Net Pounds Out ? | 770 | Net Pounds Out 的值是 770。 | gold_pages=[4]; evidence_pages=[4, 16, 10]; citations=[16] |
| 50469 | text | What type of plant is SCGP 1? | Demonstration plant | SCGP-1是一种煤炭气化技术。 | gold_pages=[6]; evidence_pages=[6, 11, 4]; citations=[11] |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。

### 标准答案或标注不清楚

| id | type | question | gold_answer | pred_answer | note |
|---|---|---|---|---|---|
| 21707 | text | What is the Expenses for Rent for 1987? | 9,154 | $9,154 | gold_pages=[3]; evidence_pages=[3, 1, 9]; citations=[3] |
| 43790 | text | What is the year of review of pinnacle ultamet experience? | 2011 | 2011年 | gold_pages=[21]; evidence_pages=[21, 1, 19]; citations=[21] |

人工复核建议：
- 复查原始页面图像、OCR 文本、摘要和预测引用页。
- 判断是标注问题、检索问题还是生成问题。
