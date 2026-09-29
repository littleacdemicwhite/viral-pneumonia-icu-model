# Streamlit 部署包（2026-09-23 重建，风险分层 0.2/0.4 与论文对齐）

原部署账号已丢失，采用全新部署。本文件夹 6 个文件一起上传到新仓库即可：

| 文件 | 说明 |
|---|---|
| app.py | 主程序（相对路径，风险分层 <20% 低 / 20–40% 中 / >40% 高） |
| medical_risk_model.pkl | ET 模型（确定性复算，AUC 0.859/0.930 与论文一致，5.2 MB） |
| medical_risk_model_features.pkl | 特征顺序（备用） |
| medical_risk_model_explainer.pkl | SHAP 解释器（7.2 MB） |
| requirements.txt | 只钉 scikit-learn==1.9.0 与 shap==0.52.0（pickle 关键），其余让云端自解 |
| README.md | 本说明（可不传） |

## 全新部署步骤（原账号丢失，2026-09-24 确定）

1. GitHub 登录或注册 → 新建 **Public** 仓库 `viral-pneumonia-icu-model`（不勾选 README/gitignore）。
2. 仓库页 → Add file → Upload files → 把本文件夹全部文件拖入 → Commit changes。
3. 打开 https://share.streamlit.io → 用 GitHub 账号登录（无需单独注册）。
4. New app → 选仓库 `viral-pneumonia-icu-model` / 分支 main / 主文件 app.py → Deploy。
   Advanced 里 Python 版本保持默认（最新）；无需填 Secrets。
5. 首次构建约 5–10 分钟。部署完成后访问 `<app名>.streamlit.app`：
   默认值点 Predict，应出现 "Model loaded successfully"，概率 + 蓝黄红三档风险
   （Medium Risk 出现在 0.2–0.4 区间即新版；旧版是 0.3–0.7）。
6. 新 URL 与旧 URL 不同：旧 app 已随账号丢失且为私有状态；稿件正文已无裸 URL，
   如需把新 URL 写进 Web application 段，部署完成后告知 Claude 同步改稿并重跑 TRIPOD 链。

