# G1 Dex3 Short Training Checkpoint

GitHub rejects ordinary files larger than 100MB. The short-training checkpoint is split into parts:

```text
Humanoid.pth.part-00
Humanoid.pth.part-01
Humanoid.pth.part-02
```

Restore it after cloning:

```bash
bash checkpoints/g1_dex3_short_train_20260717_174807/restore.sh
```

This recreates:

```text
output/g1_dex3_short_train_20260717_174807/Humanoid_17-17-48-09/nn/Humanoid.pth
```
