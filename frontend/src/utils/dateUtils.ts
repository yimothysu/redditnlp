export const months: Record<string, string> = {
  "01": "January",
  "02": "February",
  "03": "March",
  "04": "April",
  "05": "May",
  "06": "June",
  "07": "July",
  "08": "August",
  "09": "September",
  "10": "October",
  "11": "November",
  "12": "December",
};

export function formatDate(date: string, time_filter: string) {
  if (time_filter == "week") {
    return months[date.slice(0, 2)] + " " + date.slice(3, 5);
  } else if (time_filter == "year") {
    return months[date.slice(0, 2)] + " 20" + date.slice(3, 5);
  } else if (time_filter == "all") {
    return "20" + date;
  }
  return date;
}

export function createHistogram(values: number[]) {
  const minVal = 0;
  const maxVal = Math.max(...values);
  const num_bins = 30;
  const binWidth = (maxVal - minVal) / num_bins;

  const histogram = Array.from({ length: num_bins }, () => ({
    bin: 0,
    count: 0,
  }));

  values.forEach((value) => {
    const bin_index = Math.min(
      Math.floor((value - minVal) / binWidth),
      num_bins - 1
    );
    histogram[bin_index].count += 1;
  });

  return histogram.map((d, i) => ({
    bin: (minVal + i * binWidth).toFixed(2),
    count: d.count,
  }));
}
