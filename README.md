# Leander 2: Legacy of the Stick (Amiga 1200 AGA)

![Leander 2 Concept Art](assets/leander2.png)

> **"30 years later, the legend returns."**

## 📖 The Project
**Leander 2** is a passion project to create an unofficial sequel to the 1991 Psygnosis classic *Leander*. 

Built for the **Amiga 1200 (AGA)**, this project aims to combine the nostalgia of the original with the graphical power that wasn't fully utilized back in the day, enhanced by modern **Generative AI** for asset creation.

**Goal:** To deliver the sequel we dreamed of playing when we were 13.

## 👥 Credits
- **Concept & Code:** MC
- **Co-Pilot:** Google DeepMind Agent
- **Original Inspiration:** Psygnosis / Traveller's Tales (Galahad)

## 🚀 Features (So far)
- **Engine**: Custom "Bare Metal" C engine (No OS overhead).
- **Graphics**: 256-color AGA visuals, Interleaved Bitplanes.
- **Tech Stack**: 
    - Cross-compilation via Docker (`ozzyboshi/bebbo-amiga-gcc`).
    - AI-driven asset pipeline (Python + Pillow).

- **Modern Toolchain**: Cross-compilation via Docker (`ozzyboshi/bebbo-amiga-gcc`).
- **Asset Pipeline**: Python scripts (`tools/convert_assets.py`) automatically convert PNG images into Amiga bitplane format (Interleaved) and generate C headers.
- **Copper Backgrounds**: Custom Copper lists to display high-color backgrounds.

## 🛠 Prerequisites

### 1. Docker
Used for cross-compilation.
```bash
sudo apt install docker.io
sudo docker pull ozzyboshi/bebbo-amiga-gcc
```

### 2. Python 3 + Pillow
Used for asset conversion.
```bash
python3 -m venv venv
./venv/bin/pip install Pillow
```

### 3. FS-UAE (Emulator)
Used for testing the game.
```bash
sudo apt install fs-uae
```
*Note: You will need your own Kickstart ROMs (kick31.rom) in the project root or configured in FS-UAE.*

## 🏗 Building & Running

Simply run:

```bash
make run
```

This command will:
1.  Convert assets (PNG -> Amiga C Headers) if changed.
2.  Compile the C code using Docker.
3.  Fix file permissions.
4.  Launch FS-UAE with the game.

## 📂 Project Structure

- `src/`: Source code (`main.c`, hardware definitions).
- `assets/`: Graphics files. `test_bg.png` is ground truth; headers are generated.
- `tools/`: Python helper scripts.
- `S/`: Startup-Sequence for Amiga auto-boot.
