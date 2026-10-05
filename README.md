# TokenHSI g1 version Skill Demonstrations

This project is adapted from [TokenHSI](https://github.com/liangpan99/TokenHSI).


## Trajectory Skills（AMP）

The traj ckpt support a compelte walking strategy as follows:

<table>
  <tr>
    <td align="center"><strong>Traj</strong></td>
    <td align="center"><strong>In-place Turning</strong></td>
    <td align="center"><strong>Walk-Stop-Walk</strong></td>
  </tr>
  <tr>
    <td>
      <a href="./outputs/traj.mp4"><img src="./outputs/traj.gif" alt="Traj preview" width="280"></a>
    </td>
    <td>
      <a href="./outputs/traj_原地转弯.mp4"><img src="./outputs/traj_原地转弯.gif" alt="In-place turning preview" width="280"></a>
    </td>
    <td>
      <a href="./outputs/traj_走-停-走.mp4"><img src="./outputs/traj_走-停-走.gif" alt="Walk-stop-walk preview" width="280"></a>
    </td>
  </tr>
</table>

## Other three Skills（AMP）


<table>
  <tr>
    <td align="center"><strong>Sit</strong></td>
    <td align="center"><strong>Climb</strong></td>
    <td align="center"><strong>Carry</strong></td>
  </tr>
  <tr>
    <td>
      <a href="./outputs/sit.mp4"><img src="./outputs/sit.gif" alt="Sit preview" width="280"></a>
    </td>
    <td>
      <a href="./outputs/climb.mp4"><img src="./outputs/climb.gif" alt="Climb preview" width="280"></a>
    </td>
    <td>
      <a href="./outputs/carry.mp4"><img src="./outputs/carry.gif" alt="Carry preview" width="280"></a>
    </td>
  </tr>
</table>

## Whole Pipeline
Connect to qwen-3-vl-flash,put g1 in Gibson scene dataset

Give the command in terminal:"find the bridge and navigate in front of it."

<p align="center">
  <a href="./outputs/whole_pipeline.mp4"><img src="./outputs/whole_pipeline.gif" alt="Whole pipeline preview" width="860"></a>
</p>
