#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nhc/types.hpp"

TEST_CASE("Eigen 5.0.x is linked and functional", "[smoke]") {
  REQUIRE(EIGEN_MAJOR_VERSION == 5);
  REQUIRE(EIGEN_MINOR_VERSION == 0);
  REQUIRE(EIGEN_PATCH_VERSION == 1);

  Eigen::Matrix<double, 5, 5> I = Eigen::Matrix<double, 5, 5>::Identity();
  REQUIRE(I.trace() == Catch::Approx(5.0));
}

TEST_CASE("HestonParams Feller condition helper", "[smoke]") {
  nhc::HestonParams ok{0.04, 2.0, 0.04, 0.3, -0.7};
  REQUIRE(ok.satisfies_feller());

  nhc::HestonParams violated{0.04, 1.0, 0.04, 0.8, -0.7};
  REQUIRE_FALSE(violated.satisfies_feller());
}
