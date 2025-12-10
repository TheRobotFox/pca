#include "stb_image.h"
#include <cstddef>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <raylib.h>
#include <sys/types.h>

using M = Eigen::MatrixXd;
using V = Eigen::VectorXf;

auto load_image(std::string_view file) -> std::optional<V> {

  int w, h, c;
  unsigned char *data = stbi_load(file.data(), &w, &h, &c, 1);
  if (data == nullptr) {
    std::cerr << "Failed to load image: " << file << "\n";
    return {};
  }

  V vec(w * h);

  for (auto i = 0; i < w * h; i++) {
    vec(i) = data[i];
  }
  stbi_image_free(data);
  return vec;
}
auto load_images(std::string_view directory_path) -> std::optional<M> {

  std::vector<V> images;

  for (auto f : std::filesystem::directory_iterator(directory_path)) {
    if (!f.is_regular_file())
      continue;
    if (auto res = load_image(f.path().string()))
      images.push_back(*res);
  }

  M res(images.size(), images[0].size());

  for (auto [r, row] : std::views::enumerate(images))
    res.col(r) = row;

  return res;
}
auto train(const M &&imgs, auto &pcs, auto &svals, float acc) {
	auto svd = imgs.bdcSvd(Eigen::ComputeThinV);
	pcs = svd.matrixV();
	if (acc >= 1)
		svals = svd.singularValues();
	else {
		auto normalized = svd.singularValues() / svd.singularValues().sum();
		float total = 0;
		int k = 0;
		while (total < acc)
			total += normalized(k++);
		svals = svd.singularValues()(Eigen::seq(0, k));
	}
}


auto main() -> int {
	auto data_set = load_images("train")
}
