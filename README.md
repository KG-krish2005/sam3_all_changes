# SAM3 Memory-Optimized Inference

This repository contains an optimized version of the **Segment Anything Model 3 (SAM3)**, specifically modified to handle long video sequences without crashing GPU or System RAM.

🚀 **Key Improvements**

### 1. Lazy Video Loading (Memory Efficiency)
In the original SAM3, the entire video had to be loaded into memory before processing began. This caused significant RAM bottlenecks for long, high-resolution clips.

* **How it works:** I implemented the [AsyncVideoLoader class](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/io_utils.py#L28). This class emulates a standard list of frames; instead of holding all frames in memory, it loads specific frames via index only when accessed.
* **Integration:** I [modified the function](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/io_utils.py#L299C5-L299C48) `load_video_frames_from_video_file_using_cv2` to utilize this loader. 
* **Legacy Support:** The [original sam3 version](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/io_utils.py#L314) of this function has been renamed and preserved for reference.

### 2. GPU Memory Management (Inference Pruning)
Standard SAM3 keeps historical data for every person and every frame, causing VRAM usage to grow linearly over time until the GPU crashes.

* **The Fix:** I modified the [_run_single_frame_inference](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/sam3_video_inference.py#L357) function to implement a pruning mechanism.
* **Logic:** I defined a variable [keep_history_len=10](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/sam3_video_inference.py#L404C13-L404C35). Using the [removal logic](https://github.com/KG-krish2005/sam3_all_changes/blob/08f6899aaeb3b81f34ecd007ac0fece89c88018a/sam3/sam3/model/sam3_video_inference.py#L403) added here, any tracking information about frames older than 10 frames is automatically purged from the GPU.

---

## 🛠 Setup and Environment

To ensure the code runs with the correct dependencies, use the provided requirement.txt file:


---

## 🚀 How to Run

All core logic is located within the `sam/` folder. You can use the provided bash script to trigger the inference.

### Using the Bash Script
Run the following script located in the `sam` folder:
```bash
bash bashtorun.sh
```
Inside this script, you can specify:
1. The path to the input video folder (e.g., within `small_testcase/`) or a specific video.
2. Which Python script to execute.

### Available Scripts
There are three Python scripts available depending on your monitoring needs:

| Script | Functionality |
| :--- | :--- |
| `sam3_mrunal.py` | Standard inference. |
| `sam3_piyush.py` | Inference with real-time GPU usage printed to the terminal. |
| `sam3_piyush_logging.py` | Inference with GPU usage written to `sam3_run.log`. |

> [!IMPORTANT]
> **Note on Logging:** The `sam3_piyush_logging.py` script **appends** data to `sam3_run.log`. If you want a fresh log for a new run, please delete the existing `sam3_run.log` file before starting.

---

## 📂 Project Structure
* `sam/`: Contains the main inference scripts and `bashtorun.sh`.
* `small_testcase/`: Subfolders containing input video sequences.
* `environment.yml`: Precise Conda environment configuration.
