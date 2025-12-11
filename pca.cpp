#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cstddef>
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <raylib.h>
#include <sys/types.h>

using M = Eigen::MatrixXf;
using V = Eigen::VectorXf;
using Eigen::Ref;

auto load_image(std::string_view file, int &w, int &h) -> std::optional<V> {

	int c;
	unsigned char *data = stbi_load(file.data(), &w, &h, &c, 1);
	if (data == nullptr) {
		std::cerr << "Failed to load image: " << file << "\n";
		return {};
	}

	V vec(w * h);

	for (auto i = 0; i < w * h; i++) {
		vec(i) = data[i] / 255.0F;
	}
	stbi_image_free(data);
	return vec;
}
auto load_image(std::string_view file) -> std::optional<V> {
	int w, h;
	return load_image(file, w, h);
}
auto load_images(std::string_view directory_path, int &w, int &h)
    -> std::optional<M> {

	std::vector<V> images;

	for (auto f :
	     std::filesystem::recursive_directory_iterator(directory_path)) {
		if (!f.is_regular_file()) continue;
		if (auto res = load_image(f.path().string(), w, h))
			images.push_back(*res);
	}

	M res(images.size(), images[0].size());

	for (auto [r, row] : std::views::enumerate(images)) {
		res.row(r) = row;
	}

	return res;
}
auto train(Ref<const M> imgs, M &Q, V &svals, V &mean, float acc) {
	// normalize
	mean = imgs.colwise().mean();

	auto svd         = imgs.bdcSvd<Eigen::ComputeThinV>();
	const auto &S    = svd.singularValues();
	const auto &Vmat = svd.matrixV();
	if (acc == 1.0f) {
		svals = S;
		Q     = Vmat;
	} else if (acc < 1.0f) {
		Eigen::VectorXf normalized = S / S.sum();
		float total                = 0;
		int k                      = 0;

		while (k < normalized.size() && total < acc)
			total += normalized(k++);
		svals = S.head(k);
		Q     = Vmat.leftCols(k);
	} else {
		svals = S.head((int)acc);
		Q     = Vmat.leftCols((int)acc);
	}
}

#define MULT 150
auto main(int argc, char **argv) -> int {
	M data_set;

	int w, h;
	float acc = strtof(argv[2], NULL);
	if (auto res = load_images(argv[1], w, h))
		data_set = *res;
	else
		return -1;

	M Q;
	V mean, S;
	train(data_set, Q, S, mean, acc);
	M QT = Q.transpose();

	int selected_img = 0;
	V projected      = QT * (data_set.row(selected_img).transpose() - mean);
	V reconstruct;

	auto sliders = projected.rows();

	SetConfigFlags(FLAG_WINDOW_RESIZABLE);
	InitWindow(800, 600, "PCA Test");
	
	SetTargetFPS(24);

	uint8_t *data = new uint8_t[w * h];

	Image screenImage = {.data    = data,
	                     .width   = w,
	                     .height  = h,
	                     .mipmaps = 1,
	                     .format  = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE};

	Texture tex = LoadTextureFromImage(screenImage);

	std::optional<int> gripped;
	bool redraw = true;

	std::optional<float> target_time;
	V target;
	V current = projected;

	while (!WindowShouldClose()) {
		Vector2 pos = GetMousePosition();
		int screen_w = GetScreenWidth();
		int screen_h = GetScreenHeight();
		int slider_height = (screen_h / sliders);
		

		if (IsKeyPressed(KEY_N)) {
			selected_img++;
			if (selected_img >= data_set.rows()) selected_img = 0;
			target  = QT * (data_set.row(selected_img).transpose() -
			                mean);
			current = projected;

			target_time = 0;
		} else if (IsKeyPressed(KEY_P)) {
			selected_img--;
			if (selected_img < 0)
				selected_img = data_set.rows() - 1;
			target  = QT * (data_set.row(selected_img).transpose() -
			                mean);
			current = projected;
			target_time = 0;
		}

		if (target_time) {
			if (target_time >= 1) {
				projected   = target;
				target_time = {};
			} else {
				projected = current * (1 - *target_time) +
				            target * *target_time;
				target_time = *target_time + 0.1;
			}
			redraw = true;
		}

		if (IsMouseButtonUp(MOUSE_BUTTON_LEFT)) gripped = {};

		if (gripped) {
			projected(*gripped) = (pos.x / screen_w - .5) * MULT;
			redraw              = true;
		} else if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
			auto slider = (int)((pos.y) / slider_height);
			if (slider >= sliders) continue;
			gripped = slider;
		}

		if (redraw) {
			redraw      = false;
			reconstruct = (Q * projected) + mean;
			for (int i = 0; i < w * h; i++)
				data[i] = reconstruct(i) * 255;
			UpdateTexture(tex, data);
		}

		BeginDrawing();
		ClearBackground(BLACK);

		for (int i = 0; i < sliders; i++) {
			int y = i * slider_height;
			int width = projected(i) * screen_w / MULT;
			if (width > 0)
				DrawRectangle(screen_w / 2, y+slider_height/10, width, slider_height*0.8F,
				              GREEN);
			else
				DrawRectangle(screen_w / 2 + width, y+slider_height/10, -width, slider_height*0.8F,
				              GRAY);
		}
		float x_offset = screen_w/2 - (w/2)*(screen_h / h);
		DrawTextureEx(tex, {x_offset, 0}, 0,
		              (float)screen_h / h,
		              {255, 255, 255, 200});

		EndDrawing();
	}
}
