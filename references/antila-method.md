# Antila/PySleep固定方法

## 来源

- Repository: https://github.com/tortugar/Lab
- Commit: bcb8dae1594e64a511545e34f6050e2a417c1f45
- File: PySleep/sleepy.py
- SHA256: da7ad9b5a771362cc529cca98a19879837337d4e40969f684067bcacc0148a04

上游仓库未显示许可证，因此本skill不复制sleepy.py。用户手动准备本地checkout，
运行前由validate_antila_source()核对commit和文件哈希。

## 时间轴

作者频谱使用5秒窗和2.5秒步长。作者窗数为：

    2 * ceil(samples / (5 * sampling_rate)) - 1

项目主轴只保留能严格对应的窗口。作者额外的零填充尾窗单独保存。

## 校准与保护状态

默认把项目Valid窗索引传入作者use_idx。Artifact和Boundary_Unscored不参与
校准，并在作者分类后由项目外层恢复。Uncertain保持为独立状态。

配对记录只有在用户确认配对关系并批准后，才允许pooled Valid校准；该模式是
项目扩展，必须在配置和报告中明确记录。

## 输出

- Antila分期逐窗结果
- 作者额外尾窗
- 阶段时长与比例
- 睡眠片段和状态转换
- 作者源码版本、输入、参数、运行记录与SHA256

