# 机器人综合实验报告

编译方式：

```bash
cd report
latexmk -xelatex main.tex
```

如果没有 `latexmk`，可使用：

```bash
cd report
xelatex main.tex
xelatex main.tex
```

报告采用多文件结构，`main.tex` 负责全局格式和章节引入，各实验章节位于 `chapters/` 目录。
