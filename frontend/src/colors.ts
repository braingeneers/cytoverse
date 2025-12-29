// Generate distinct colors for labels + one extra color for test points
// This function generates a set of colors that are evenly spaced in the HSL color space,
// avoiding bright red (hue 0) to ensure good visibility and distinction between labels.
// It returns an array of hex color strings.
// The number of labels is passed as an argument, and it generates colors starting from hue 30 (yellow) to hue 360 (red), ensuring that the colors are visually distinct and not too bright or saturated.
// The colors are generated in HSL format and converted to hex format for use in the scatterplot.
export const generateLabelColors = (numLabels: number): string[] => {
  const colors: string[] = [];

  // Use HSL to generate evenly spaced colors, avoiding bright red (hue 0)
  for (let i = 0; i < numLabels; i++) {
    // Skip hue 0 (red) and start from 30 degrees to avoid bright red
    const hue = 30 + (i * 330) / numLabels;
    const saturation = 0.7;
    const lightness = 0.5;

    // Convert HSL to RGB
    const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
    const m = lightness - c / 2;

    let r: number, g: number, b: number;

    if (hue >= 0 && hue < 60) {
      r = c;
      g = x;
      b = 0;
    } else if (hue >= 60 && hue < 120) {
      r = x;
      g = c;
      b = 0;
    } else if (hue >= 120 && hue < 180) {
      r = 0;
      g = c;
      b = x;
    } else if (hue >= 180 && hue < 240) {
      r = 0;
      g = x;
      b = c;
    } else if (hue >= 240 && hue < 300) {
      r = x;
      g = 0;
      b = c;
    } else {
      r = c;
      g = 0;
      b = x;
    }

    const rHex = Math.round((r + m) * 255)
      .toString(16)
      .padStart(2, '0');
    const gHex = Math.round((g + m) * 255)
      .toString(16)
      .padStart(2, '0');
    const bHex = Math.round((b + m) * 255)
      .toString(16)
      .padStart(2, '0');

    colors.push(`#${rHex}${gHex}${bHex}`);
  }

  colors.push('#999999'); // Add gray for un-labeled query cells

  return colors;
};
