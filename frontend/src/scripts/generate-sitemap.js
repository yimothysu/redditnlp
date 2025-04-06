import { SitemapStream, streamToPromise } from "sitemap";
import { createWriteStream } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const subreddits = [
  "AskReddit",
  "politics",
  "AskOldPeople",
  "gaming",
  "science",
  "popculturechat",
  "worldnews",
  "technology",
  "100YearsAgo",
  "Feminism",
  "unpopularopinion",
  "philosophy",
  "mentalhealth",
  "teenagers",
  "AskMen",
  "AskWomen",
  "personalfinance",
  "changemyview",
  "LateStageCapitalism",
  "UpliftingNews",
];

const BASE_URL = "https://redditnlp.com";

const staticRoutes = ["/"];
const subredditRoutes = subreddits.map(
  (subreddit) => `/subreddit/${subreddit}`
);

const routes = [...staticRoutes, ...subredditRoutes];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sitemapPath = path.join(__dirname, "./../../public/sitemap.xml");

const writeStream = createWriteStream(sitemapPath);
const sitemapStream = new SitemapStream({ hostname: BASE_URL });

streamToPromise(sitemapStream)
  .then((data) => {
    writeStream.end(data);
    console.log("Sitemap generated at public/sitemap.xml");
  })
  .catch((err) => {
    console.error("Failed to generate sitemap: ", err);
  });

routes.forEach((route) => {
  sitemapStream.write({
    url: route,
    changefreq: "weekly",
    priority: route === "/" ? 1.0 : 0.8,
  });
});

sitemapStream.end();
