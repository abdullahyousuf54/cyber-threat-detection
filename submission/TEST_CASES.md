# Test Cases

| ID | Test | Expected Result |
|---|---|---|
| TC01 | Start dashboard | Dashboard opens without application error |
| TC02 | Load trained models | Available model status is shown |
| TC03 | Run sample threat detection | Predicted class and confidence are displayed |
| TC04 | Calculate risk | LOW/MEDIUM/HIGH/CRITICAL risk is returned |
| TC05 | Generate response | One of MONITOR/ALERT/BLOCK/ISOLATE is recommended |
| TC06 | Submit Accept feedback | Feedback is stored and learning update is applied |
| TC07 | Submit Reject feedback | Feedback is stored and policy is updated |
| TC08 | Run anomaly detection | Anomaly/normal indication is displayed |
| TC09 | Run evaluation (`python evaluate_step4.py`) | Metrics JSON/evaluation artifacts are generated |
| TC10 | Run tuning analysis (`python step7_analysis.py`) | Class distribution, tuned RF and feature importance files are generated |
