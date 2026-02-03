#pragma once

namespace coin
{

  /** Convert pixel diameter to mm using scale ratio (mm per pixel). */
  double pixel_diameter_to_mm(double diameter_px, double ratio_px_to_mm);

  /** Convert diameter in mm to radius in pixels. */
  int diameter_mm_to_radius_px(double diameter_mm, double ratio_px_to_mm);

}
