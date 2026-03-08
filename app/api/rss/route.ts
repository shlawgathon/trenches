import { NextResponse } from "next/server";
import { getAllSources } from "@/src/lib/data-sources/registry";

export const dynamic = "force-dynamic";
export const revalidate = 0;

type RssItem = {
  title: string;
  originalTitle: string | null;
  link: string;
  translateUrl: string | null;
  source: string;
  agent: string;
  pubDate: string;
  bootstrapOnly: true;
};

// Sources whose content is not in English
const NON_ENGLISH_SOURCE_IDS = new Set([
  "iran-bbc-persian",
  "iran-fars-news",
]);

// Translate text to English using Google's free translate endpoint
async function translateToEnglish(text: string): Promise<string> {
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=${encodeURIComponent(text)}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return text;
    const data = await res.json();
    // Response format: [[["translated","original",...],...],...] 
    const translated = (data?.[0] ?? [])
      .map((seg: [string]) => seg?.[0] ?? "")
      .join("");
    return translated || text;
  } catch {
    return text;
  }
}

// Simple XML tag extractor (no dependency needed)
function extractTag(xml: string, tag: string): string {
  const regex = new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]></${tag}>|<${tag}[^>]*>([^<]*)</${tag}>`);
  const match = xml.match(regex);
  return (match?.[1] ?? match?.[2] ?? "").trim();
}

function extractItems(xml: string): Array<{ title: string; link: string; pubDate: string }> {
  const items: Array<{ title: string; link: string; pubDate: string }> = [];
  const itemRegex = /<item>([\s\S]*?)<\/item>/g;
  let match;
  while ((match = itemRegex.exec(xml)) !== null && items.length < 5) {
    const block = match[1];
    const title = extractTag(block, "title");
    const link = extractTag(block, "link");
    const pubDate = extractTag(block, "pubDate");
    if (title && link) {
      items.push({ title, link, pubDate });
    }
  }
  return items;
}

// Pick a few key RSS sources per agent for a fast, diverse feed
const PRIORITY_IDS = [
  "us-reuters-us",
  "us-politico",
  "us-npr-news",
  "israel-haaretz",
  "israel-jerusalem-post",
  "israel-times-of-israel",
  "israel-middle-east-eye",
  "iran-iran-international",
  "iran-fars-news",
  "iran-bbc-persian",
  "hezbollah-reuters",
  "hezbollah-aljazeera",
  "hezbollah-lorient-lejour",
  "hezbollah-naharnet",
  "gulf-arabian-business",
  "gulf-arab-news",
  "gulf-the-national-gcc",
  "oversight-unhcr-feed",
];

export async function GET() {
  const allSources = getAllSources();
  const rssSources = allSources.filter(
    (s) =>
      s.kind === "rss" &&
      s.endpoint.kind === "url" &&
      PRIORITY_IDS.includes(s.id)
  );

  const results: RssItem[] = [];

  const fetches = rssSources.map(async (source) => {
    try {
      if (source.endpoint.kind !== "url") return;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(source.endpoint.url, {
        signal: controller.signal,
        headers: { "User-Agent": "Trenches/1.0 RSS Reader" },
      });
      clearTimeout(timeout);
      if (!res.ok) return;
      const xml = await res.text();
      const items = extractItems(xml);
      const needsTranslation = NON_ENGLISH_SOURCE_IDS.has(source.id);

      for (const item of items) {
        let title = item.title;
        let originalTitle: string | null = null;

        if (needsTranslation) {
          originalTitle = item.title;
          title = await translateToEnglish(item.title);
        }

        results.push({
          title,
          originalTitle,
          link: item.link,
          translateUrl: needsTranslation
            ? `https://translate.google.com/translate?sl=auto&tl=en&u=${encodeURIComponent(item.link)}`
            : null,
          source: source.name,
          agent: source.agentId,
          pubDate: item.pubDate || new Date().toISOString(),
          bootstrapOnly: true,
        });
      }
    } catch {
      // Feed unavailable, skip silently
    }
  });

  await Promise.allSettled(fetches);

  // Sort by pubDate descending
  results.sort((a, b) => {
    const da = new Date(a.pubDate).getTime() || 0;
    const db = new Date(b.pubDate).getTime() || 0;
    return db - da;
  });

  return NextResponse.json(
    { items: results.slice(0, 40) },
    {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    },
  );
}
