#include "stb_image.h"
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <raylib.h>
#include <sys/types.h>

using Matrix = Eigen::MatrixXd;

auto load_images(std::string_view directory_path) -> std::optional<Matrix> {

	std::vector<std::unique_ptr<uint8_t>> images;
	int expect_w = 0, expect_h = 0;

	for (const auto &entry :
	   std::filesystem::directory_iterator(directory_path)) {
		if (!entry.is_regular_file())
			continue;

		auto path = entry.path().string();

		// Attempt to load the image
		int w, h, c;
		unsigned char *data = stbi_load(path.c_str(), &w, &h, &c, 1);
		if (data == nullptr) {
			std::cerr << "Failed to load image: " << path << "\n";
			continue;
		}
		if (expect_h < 0 || expect_w < 0) {
			expect_h = h;
			expect_w = w;
		} else if (expect_h != h || expect_w != w)
			return {};

		images.emplace_back(data);
    }

    Matrix res(images.size(), expect_h * expect_w);
    for (auto [i, d] : std::views::enumerate(images)) {
		res()
		
    }
	
	return true;
}

auto train()


auto main() -> int {
  Matrix m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}
