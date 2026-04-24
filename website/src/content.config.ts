import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blog = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    publishedAt: z.coerce.date(),
    ntiScore: z.number(),
    ntiZone: z.string(),
    ntiZonePrev: z.string(),
    zoneChanged: z.boolean().default(false),
    confidence: z.number(),
    nifty50Price: z.number().nullable().optional(),
    topDrivers: z.array(z.string()).default([]),
    topStocks: z.array(z.string()).default([]),
    blogType: z.enum([
      "market_open",
      "mid_session",
      "market_close",
      "post_market",
      "overnight",
    ]).default("mid_session"),
  }),
});

export const collections = { blog };
