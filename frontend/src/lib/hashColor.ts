export function hashColor(str: string) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  const hue = Math.abs(hash % 360); // 0-360
  const saturation = 25 + Math.abs((hash >> 8) % 35); // 25-60%
  const lightness = 85 + Math.abs((hash >> 16) % 10); // 85-95%

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}
