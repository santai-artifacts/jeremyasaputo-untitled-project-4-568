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

const SYSTEM = `You are a Mad Libs author. Given a THEME from the user, write a short, funny, family-friendly Mad Libs story (about 90-140 words) whose plot and setting are clearly centered on that theme.

Return ONLY valid JSON, no markdown fences, matching exactly:
{
  "title": "a short punchy title related to the theme",
  "template": "the full story text with blanks written as tokens [[1]], [[2]], ... in reading order",
  "blanks": [{ "id": 1, "label": "part of speech or hint" }, ...]
}

Rules:
- Use between 8 and 12 blanks.
- Token ids are sequential starting at 1 and each id appears exactly once in the template, in order.
- Each blank label is a concrete grammar hint the player can fill, e.g. "adjective", "plural noun", "verb ending in -ing", "silly name", "body part", "exclamation", "number".
- Vary the parts of speech. Keep it whimsical and lightly absurd.
- Never put a real word inside the [[ ]] tokens — only the number.
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
  return { title: data.title, template: data.template, blanks };
}

async function generate(prompt: string) {
  const msg = await ai.messages.create({
    model: MODEL,
    max_tokens: 2048,
    system: SYSTEM,
    messages: [
      {
        role: "user",
        content: `THEME: ${prompt}\n\nWrite the Mad Libs story JSON now.`,
      },
    ],
  });
  const text = msg.content
    .map((b: any) => (b.type === "text" ? b.text : ""))
    .join("");
  return validate(extractJson(text));
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

        // One retry — the model occasionally returns almost-JSON.
        let data;
        try {
          data = await generate(prompt);
        } catch {
          data = await generate(prompt);
        }
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
