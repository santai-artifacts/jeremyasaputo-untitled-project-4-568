import Anthropic from "@anthropic-ai/sdk";

const publicDir = `${import.meta.dir}/public`;

const ai = new Anthropic({
  baseURL: process.env.SANTAI_AI_BASE_URL,
  apiKey: process.env.SANTAI_AI_TOKEN,
});

const MODEL = "anthropic-claude-bedrock4.5-haiku";

// --- Types the client relies on ---
// {
//   title: string,
//   template: string,        // contains tokens like [[1]], [[2]]
//   blanks: [{ id: number, label: string }]  // label = human hint, e.g. "plural noun"
// }

const SYSTEM = `You are a world-class Mad Libs author. Given a THEME, write a genuinely FUNNY, family-friendly Mad Libs story whose plot and setting are clearly centered on that theme.

Return ONLY valid JSON, no markdown fences, matching exactly:
{
  "title": "a short punchy title related to the theme",
  "template": "the full story with blanks written as tokens [[1]], [[2]], ... in reading order",
  "blanks": [{ "id": 1, "label": "part-of-speech hint" }, ...]
}

STRUCTURE:
- Exactly 5 paragraphs, separated by a blank line (\\n\\n). Give it a real arc: setup, rising trouble, a wild twist, a climax, a punchy payoff ending.
- Aim for 140-200 words total across the paragraphs.
- Use between 10 and 14 blanks, spread across all 5 paragraphs.

HUMOR (this matters most):
- Be actually funny: absurd juxtapositions, comic escalation, a surprising twist, and a punchline in the final paragraph. Not just "silly" — genuinely amusing.

GRAMMAR — every blank MUST read correctly no matter what the player types. This is critical:
- The sentence around a token must be grammatical for ANY word of that label. Write the sentence so the label fits naturally.
- "adjective" is ONLY ever placed directly before a noun (e.g. "a [[x]] hat", "the [[x]] dog") — NEVER after "in", "was", or standing alone. If a slot follows "in the", "was", "into", or ends a phrase, it must be a NOUN, not an adjective.
- "noun" / "plural noun" go where a thing belongs; "verb" where an action belongs; "verb ending in -ing" only after "was/were/started" or as a gerund; "number" only where a count fits; "exclamation" only inside quotes as an interjection.
- Do NOT stack two number-style blanks together. Do NOT label a blank "adjective" if the fix would be a noun.
- Re-read each sentence with a plausible filler word for that label and confirm it is grammatical before finalizing.

TOKENS:
- Token ids are sequential from 1, each id appears exactly once, in reading order.
- Never put a real word inside [[ ]] — only the number.
- Keep it clean and suitable for all ages.`;

function extractJson(text: string): any {
  // Model should return raw JSON, but be forgiving of fences / stray prose.
  let t = text.trim();
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) t = fence[1].trim();
  const start = t.indexOf("{");
  const end = t.lastIndexOf("}");
  if (start !== -1 && end !== -1) t = t.slice(start, end + 1);
  return JSON.parse(t);
}

function validate(data: any) {
  if (!data || typeof data !== "object") throw new Error("bad shape");
  if (typeof data.title !== "string" || typeof data.template !== "string")
    throw new Error("missing title/template");
  if (!Array.isArray(data.blanks) || data.blanks.length === 0)
    throw new Error("missing blanks");
  const blanks = data.blanks.map((b: any, i: number) => ({
    id: Number(b.id ?? i + 1),
    label: String(b.label ?? "word"),
  }));
  // Ensure every blank id is actually referenced in the template.
  for (const b of blanks) {
    if (!data.template.includes(`[[${b.id}]]`))
      throw new Error(`template missing token for blank ${b.id}`);
  }

  // Deterministic grammar linter: a blank right after a preposition cannot be
  // an adjective ("the kitchen in ___" needs a noun, not "messy"). Haiku makes
  // this exact mistake, so we repair the label rather than trust the model.
  const PREPS = new Set([
    "in", "into", "of", "with", "from", "to", "at", "on", "onto",
    "for", "by", "about", "over", "under", "through", "like", "as",
  ]);
  for (const b of blanks) {
    if (!/adjective/i.test(b.label)) continue;
    const re = new RegExp(`(\\w+)\\s+\\[\\[${b.id}\\]\\]`);
    const m = data.template.match(re);
    if (m && PREPS.has(m[1].toLowerCase())) {
      b.label = "noun";
    }
  }

  // Structural quality gates — reject so the caller regenerates.
  const paragraphs = data.template.split(/\n{2,}/).map((p: string) => p.trim()).filter(Boolean);
  if (paragraphs.length < 4)
    throw new Error(`only ${paragraphs.length} paragraphs`);
  if (blanks.length < 9)
    throw new Error(`only ${blanks.length} blanks`);

  return { title: data.title, template: data.template, blanks };
}

// A full worked example teaches Haiku the exact shape, paragraph count, and
// (crucially) grammatical slot placement far better than rules alone.
const EXAMPLE = JSON.stringify({
  title: "The Haunted Laundromat",
  template:
    "It was a [[1]] Tuesday when Greg discovered his neighborhood laundromat was haunted by a [[2]] ghost. The ghost only appeared when someone loaded exactly [[3]] socks into the dryer.\n\nGreg didn't believe it, so he marched inside carrying a basket of [[4]]. The moment the machine started [[5]], the lights flickered and a voice moaned from behind the change dispenser.\n\n\"[[6]]!\" shouted the ghost, floating out wearing a [[7]] bathrobe. It demanded that Greg fold every towel using only his [[8]]. Greg, being extremely [[9]], agreed immediately.\n\nAs a reward, the ghost taught Greg the ancient secret of removing [[10]] stains, a technique passed down for [[11]] generations. They became the unlikeliest of friends.\n\nNow, every Tuesday, Greg and the ghost host a [[12]] laundry party, and it is officially the most [[13]] event in town.",
  blanks: [
    { id: 1, label: "adjective" },
    { id: 2, label: "adjective" },
    { id: 3, label: "number" },
    { id: 4, label: "plural noun" },
    { id: 5, label: "verb ending in -ing" },
    { id: 6, label: "exclamation" },
    { id: 7, label: "color" },
    { id: 8, label: "body part (plural)" },
    { id: 9, label: "adjective" },
    { id: 10, label: "plural noun" },
    { id: 11, label: "number" },
    { id: 12, label: "adjective" },
    { id: 13, label: "adjective" },
  ],
});

async function generateOnce(prompt: string) {
  const msg = await ai.messages.create({
    model: MODEL,
    max_tokens: 2048,
    system: SYSTEM,
    messages: [
      {
        role: "user",
        content:
          `Here is a perfect example of the JSON for the theme "a haunted laundromat". ` +
          `Notice: exactly 5 paragraphs, 13 blanks, and every blank reads grammatically ` +
          `(adjectives sit before nouns, "in/into" is never followed by an adjective):\n\n` +
          EXAMPLE,
      },
      {
        role: "assistant",
        content: "Understood — 5 paragraphs, grammatically sound blanks, JSON only.",
      },
      {
        role: "user",
        content: `Now write a NEW Mad Libs story JSON for this theme: ${prompt}`,
      },
    ],
  });
  const text = msg.content
    .map((b: any) => (b.type === "text" ? b.text : ""))
    .join("");
  return validate(extractJson(text));
}

async function generate(prompt: string) {
  let lastErr: unknown;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      return await generateOnce(prompt);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

const server = {
  port: process.env.PORT || 3000,
  async fetch(req: Request) {
    const url = new URL(req.url);

    if (url.pathname === "/api/generate" && req.method === "POST") {
      try {
        if (!process.env.SANTAI_AI_TOKEN) {
          return Response.json(
            { error: "AI is not configured in this environment." },
            { status: 503 }
          );
        }
        const body = await req.json().catch(() => ({}));
        const prompt = String(body?.prompt ?? "").trim();
        if (!prompt) {
          return Response.json({ error: "Please enter a prompt." }, { status: 400 });
        }
        if (prompt.length > 300) {
          return Response.json({ error: "Prompt is too long." }, { status: 400 });
        }

        const data = await generate(prompt);
        return Response.json(data);
      } catch (err) {
        console.error("generate failed:", err);
        return Response.json(
          { error: "Couldn't write that story. Try a different prompt." },
          { status: 500 }
        );
      }
    }

    // Static files
    let pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    const file = Bun.file(`${publicDir}${pathname}`);
    if (await file.exists()) return new Response(file);

    // SPA-ish fallback
    return new Response(Bun.file(`${publicDir}/index.html`));
  },
};

export default server;
