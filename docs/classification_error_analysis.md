# Error Analysis — Phase 2

**Model:** LogisticRegression  
**Evaluated on:** test set (120 samples)  
**Misclassified:** 13 (10.8%)

## Confusion Matrix

| True \ Predicted | email | form | invoice | receipt |
|---|---|---|---|---|
| **email** | 28 | 2 | 0 | 0 |
| **form** | 0 | 23 | 7 | 0 |
| **invoice** | 0 | 4 | 26 | 0 |
| **receipt** | 0 | 0 | 0 | 30 |

## Most Confused Class Pairs

| True Label | Predicted As | Count |
|---|---|---:|
| form | invoice | 7 |
| invoice | form | 4 |
| email | form | 2 |

## Per-Class Accuracy

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| email | 28 | 30 | 93.3% |
| form | 23 | 30 | 76.7% |
| invoice | 26 | 30 | 86.7% |
| receipt | 30 | 30 | 100.0% |

## Key Questions

**Are invoices and receipts being mixed up?**  
Invoice → Receipt: 0, Receipt → Invoice: 0

**Are forms being mistaken for invoices?**  
Form → Invoice: 7, Invoice → Form: 4

## Misclassified Examples

### rvl_test_17339.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `parry card est jackson boulevard fe chcas ken register cf 0806-015 philip mores co lto. ing ctt counrey new zealand en 482285 case lieser et al us 585122 5th vax takomte aug 6 1958 hereby instructed t`

### rvl_train_24271.png
- **True:** email  
- **Predicted:** form  
- **Snippet:** `marketing research proposal project 96-3200-09 title stexploratory executive management background executive management expressed interest observing smoker reaction three winston campaigns approved ca`

### rvl_test_02340.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `leaease read face reverse ny fs ad conon nd hose nara fefrre en rern es pre hal conse ine cnet betwen ie pars noa son tet toms wel condos ae ep 666 fifth avenue new york n.y. 10103 purchase order date`

### rvl_val_05355.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `despachse loo agentes ofictalos ia propiedad industriat sres elzaburu philip morris incorporate richmond ee uu nous ct igo angst 28 pnpses ee october 9 '1982 remittance letters patent law department-p`

### rvl_train_09780.png
- **True:** invoice  
- **Predicted:** form  
- **Snippet:** `classified material receipt nonte coies clossitcaton secret confidential deliver legs fhe 50643 6632`

### rvl_train_16011.png
- **True:** email  
- **Predicted:** form  
- **Snippet:** `tobacco merchants association daily executive summary 2 today 's news highlights 42/26/2000 world 2001 philip mons plans fo increase cigarette proton rusia 10-15 pervert 446 billion cigarettes almost `

### rvl_train_05009.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `409/12 91 15:29 332 272 51289 ss inbifo koein eo 2264 crc contract research center pvaaisrar number folowing pages 2 van veer de m. merckx por frau b. viol ven pe date projektinformationen 36026 score`

### rvl_train_21478.png
- **True:** invoice  
- **Predicted:** form  
- **Snippet:** `pot 3 81440667`

### rvl_train_22292.png
- **True:** invoice  
- **Predicted:** form  
- **Snippet:** `mat bg 31834 /9nity3a31s34 bead banhida aimee tb oa oz oa yve geen 7a uses saoven| oma pir penance ~gnguh sunt tho soo sie iw varwaa wain3auz ze6t siow 30 nii sunpp 7 fess qinyn 70 inh 5 aons/samn mbl`

### rvl_train_18391.png
- **True:** invoice  
- **Predicted:** form  
- **Snippet:** `gaa executiy vip yuda \4 ceb 49842 ofen wld 70 pek ote officia taxi rec x 8 8 py 3 g 8 8`

### rvl_train_19783.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `repair return requisition service class 2 date 10/05/93 suggested supplier emc12 1 dept req 0¥42780201 electric motor contracting tirequestor vargo dana hl ship 500 po deliver stockroom 411000 ul via `

### rvl_train_11271.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `imp ras eon 50090 3005`

### rvl_train_03593.png
- **True:** form  
- **Predicted:** invoice  
- **Snippet:** `outgoing malb egister crc contract research center avaaserne im neuenheimer feld 280 0-6900 heidelberg remarks sehr geehrter herr doktor fdratenberger da wir sle nach einigen versuchen nicht per telef`

