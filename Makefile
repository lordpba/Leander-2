# Docker configuration
DOCKER_IMG = ozzyboshi/bebbo-amiga-gcc
DOCKER_CMD = docker run --rm -v "$$(pwd)":/src -w /src $(DOCKER_IMG)

# Compiler settings
CC = $(DOCKER_CMD) m68k-amigaos-gcc
CFLAGS = -noixemul -m68020-60 -m68881 -O2 -Wall
LDFLAGS = -noixemul -lamiga

SRC = src/main.c
OBJ = $(SRC:.c=.o)
TARGET = Leander2

all: $(TARGET)


# Asset Generation (Generic rule for converting PNG to C headers)
ASSETS_DIR = assets
TOOLS_DIR = tools

# Dependency: Header files depend on the Python script and source PNG
$(ASSETS_DIR)/%_data.h: $(ASSETS_DIR)/%.png $(TOOLS_DIR)/convert_assets.py
	./venv/bin/python $(TOOLS_DIR)/convert_assets.py $< $(basename $<) 5

# Ensure TARGET depends on the specific background we are using
$(TARGET): $(SRC) $(ASSETS_DIR)/test_bg_data.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
	$(DOCKER_CMD) chown $(shell id -u):$(shell id -g) $@


run: $(TARGET)
	fs-uae --amiga-model=A1200 --hard-drive-0=. --progdir-is-cwd

clean:
	rm -f $(OBJ) $(TARGET)
