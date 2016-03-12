all: colour-extract-blob get-blob-positions

colour-extract-blob:
	g++ -g -ggdb `pkg-config --cflags opencv` colour-extract-blob.cpp -o colour-extract-blob `pkg-config --libs opencv`

get-blob-positions:
	g++ -g -ggdb `pkg-config --cflags opencv` get-blob-positions.cpp -o get-blob-positions `pkg-config --libs opencv`