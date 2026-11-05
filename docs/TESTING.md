# 测试方法

## 后端

从`model-rag`根目录使用项目Python环境运行：

```powershell
& '.\.venv\Scripts\python.exe' -B tools/run_tests.py
```

该入口固定使用临时数据目录和SQLite，阻止外部HTTP请求，不重建或清理业务MySQL。原单用户业务测试使用真实JWT夹具，多用户边界测试直接运行生产认证中间件和按用户检索注册表。

未安装可选论文原文、原图时，相应集成检查跳过；正文解析、引用、权限和合成夹具测试仍运行。功能测试结果不代表模型回答准确率。开发题集的标注来源及人工复核状态保存在题目字段中。

## 前端

进入`model-rag/frontend`执行：

```powershell
node --test tests/*.test.js
npm.cmd run build
```

前端检查涵盖流式协议、导航、引用、追问、账号存储分区和CSRF辅助函数。实际页面可按使用说明核对模型阅读、文献导入、连续提问及账号切换。
