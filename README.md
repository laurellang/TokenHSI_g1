# TokenHSI g1 version Skill Demonstrations

This project is adapted from [TokenHSI](https://github.com/liangpan99/TokenHSI).


## Trajectory Skills

The traj ckpt support a compelte walking strategy as follows:s

<table>
  <tr>
    <td align="center"><strong>Traj</strong></td>
    <td align="center"><strong>In-place Turning</strong></td>
    <td align="center"><strong>Walk-Stop-Walk</strong></td>
  </tr>
  <tr>
    <td>
      <video src="./outputs/traj.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/traj.mp4">Open Traj video</a>
      </video>
    </td>
    <td>
      <video src="./outputs/traj_原地转弯.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/traj_原地转弯.mp4">Open in-place turning video</a>
      </video>
    </td>
    <td>
      <video src="./outputs/traj_走-停-走.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/traj_走-停-走.mp4">Open walk-stop-walk video</a>
      </video>
    </td>
  </tr>
</table>

## Skills

other three ckpts:

<table>
  <tr>
    <td align="center"><strong>Sit</strong></td>
    <td align="center"><strong>Climb</strong></td>
    <td align="center"><strong>Carry</strong></td>
  </tr>
  <tr>
    <td>
      <video src="./outputs/sit.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/sit.mp4">Open sit video</a>
      </video>
    </td>
    <td>
      <video src="./outputs/climb.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/climb.mp4">Open climb video</a>
      </video>
    </td>
    <td>
      <video src="./outputs/carry.mp4" type="video/mp4" controls width="280">
        <a href="./outputs/carry.mp4">Open carry video</a>
      </video>
    </td>
  </tr>
</table>

## Whole Pipeline
Connect to qwen-3-vl-flash,put g1 in Gibson scene dataset

Give the command in terminal:"find the bridge and navigate in front of it."

<p align="center">
  <video src="./outputs/whole_pipeline.mp4" type="video/mp4" controls width="860">
    <a href="./outputs/whole_pipeline.mp4">Open whole-pipeline video</a>
  </video>
</p>

