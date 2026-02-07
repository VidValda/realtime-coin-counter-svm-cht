#pragma once

namespace coin
{

  double pixel_diameter_to_mm(double diameter_px, double ratio_px_to_mm);

  int diameter_mm_to_radius_px(double diameter_mm, double ratio_px_to_mm);

}
