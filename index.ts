import Anthropic from "@anthropic-ai/sdk";
import Database from "bun:sqlite";
import { mkdirSync } from "node:fs";

const publicDir = `${import.meta.dir}/public`;

const ai = new Anthropic({
  baseURL: process.env.SANTAI_AI_BASE_URL,
  apiKey: process.env.SANTAI_AI_TOKEN,
});

// --- Persistence (saved-story gallery) ---
const dbPath = process.env.DATABASE_URL || "./data/app.db";
try {
  const dir = dbPath.slice(0, dbPath.lastIndexOf("/"));
  if (dir) mkdirSync(dir, { recursive: true });
} catch {}
const db = new Database(dbPath);
db.exec(`
  CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    prompt TEXT NOT NULL,
    template TEXT NOT NULL,
    blanks TEXT NOT NULL,
    vals TEXT NOT NULL,
    created_at TEXT NOT NULL
  )
`);

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
- A VERB blank ("verb", "verb (past tense)", "verb ending in -ed") must be the ONLY verb in its clause — it IS the action. NEVER place a verb blank right before or after another verb. WRONG: "My dog [[x]] announced his plan" (two verbs → "wiggled announced"). RIGHT: "My dog [[x]] down the street" or "My dog loudly [[x]] at the mailman." If the subject already has a real verb like "announced", do not insert a verb blank next to it — make that slot an adverb or move it elsewhere.
- Do NOT stack two number-style blanks together. Do NOT label a blank "adjective" if the fix would be a noun.
- Re-read each sentence by dropping in a plausible filler word for that label. If it produces two verbs in a row, an adjective after "in/into", or any broken grammar, rewrite the sentence before finalizing.

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

  // Deterministic grammar linter #2: a verb blank sitting next to another verb
  // makes two verbs in a row ("My dog ___ announced" -> "wiggled announced").
  // Relabel it to "adverb", which always reads correctly ("proudly announced").
  const IRREGULAR = new Set([
    "ran", "went", "said", "ate", "saw", "made", "won", "took", "gave", "held",
    "flew", "drove", "sang", "told", "became", "stood", "sat", "found", "led",
    "grew", "threw", "spoke", "broke", "wore", "rode", "shook", "began",
  ]);
  const looksVerb = (w: string) => {
    const x = (w || "").toLowerCase();
    return /(?:ed|es)$/.test(x) || IRREGULAR.has(x);
  };
  for (const b of blanks) {
    if (!/verb/i.test(b.label) || /ing/i.test(b.label)) continue; // gerunds are handled by their own placement
    const re = new RegExp(`(\\w+)?\\s*\\[\\[${b.id}\\]\\]\\s*(\\w+)?`);
    const m = data.template.match(re);
    if (!m) continue;
    const before = m[1] || "";
    const after = m[2] || "";
    if (looksVerb(after) || looksVerb(before)) {
      b.label = "adverb";
    }
  }

  // Deterministic grammar linter #3: a plural-noun blank right after "a"/"an"
  // reads "a spatulas". Force it singular so the fill agrees with the article.
  // (The a/an sound itself is fixed at render time, once the word is known.)
  for (const b of blanks) {
    if (!/noun/i.test(b.label) || !/plural|nouns/i.test(b.label)) continue;
    const re = new RegExp(`(^|[\\s("'])(an?)\\s+\\[\\[${b.id}\\]\\]`, "i");
    if (re.test(data.template)) b.label = "noun";
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

// --- Funny single-word suggestions (shared by /api/word and /api/fill) ---
const FUNNY_WORD_SYSTEM =
  "You suggest the FUNNIEST possible word for a Mad Libs blank — absurd, unexpected, " +
  "delightfully random. It MUST fit the grammatical part of speech named in the hint " +
  "(a noun stays a noun, a verb stays a verb, etc.), but it does NOT need to make logical " +
  "sense in the story — the sillier the better. Family-friendly, nothing offensive. " +
  "Reply with ONLY the word (or a 2-word name/phrase if the hint asks for a name). " +
  "No quotes, no punctuation, no explanation. Lowercase unless it is a proper name.";

function sanitizeWord(raw: string, label = ""): string {
  let w = String(raw || "")
    .split("\n")[0]
    .trim()
    .replace(/^["'“‘(]+|["'”’).,!?;:]+$/g, "")
    .trim()
    .slice(0, 40);
  // Backstop against the model returning a whole phrase/clause instead of a word.
  // Names/titles may be a few words; everything else is at most two.
  const maxWords = /name|place|title|exclamation/i.test(label) ? 3 : 2;
  const tokens = w.split(/\s+/).filter(Boolean);
  if (tokens.length > maxWords) w = tokens.slice(0, maxWords).join(" ");
  return w;
}

// Cross-request memory so the same "funny" words don't get recycled every story.
const RECENT_MAX = 160;
const recentWords: string[] = [];
const recentSet = new Set<string>();
function remember(words: string[]) {
  for (const w of words) {
    const k = (w || "").trim().toLowerCase();
    if (!k || recentSet.has(k)) continue;
    recentWords.push(k);
    recentSet.add(k);
  }
  while (recentWords.length > RECENT_MAX) {
    const old = recentWords.shift();
    if (old) recentSet.delete(old);
  }
}
function recentSample(n: number): string[] {
  return recentWords.slice(-n);
}

// A random comedic "territory" nudges each fill somewhere new, fighting the
// model's tendency to reach for the same handful of favorites.
const ANGLES = [
  "kitchen gadgets", "obscure animals", "types of weather", "musical instruments",
  "old-timey jobs", "breakfast foods", "dance moves", "sea creatures",
  "office supplies", "types of hats", "garden vegetables", "board games",
  "clumsy sound effects", "fancy desserts", "camping gear", "gemstones",
  "circus acts", "types of pasta", "bugs and insects", "space stuff",
];
function randomAngle(): string {
  return ANGLES[Math.floor(Math.random() * ANGLES.length)];
}

// Pull the sentence that contains a token, with the token shown as "___".
function sentenceFor(template: string, id: number): string {
  const tok = `[[${id}]]`;
  const idx = template.indexOf(tok);
  if (idx === -1) return "";
  const left = template.slice(0, idx);
  const right = template.slice(idx + tok.length);
  const lb = Math.max(left.lastIndexOf(". "), left.lastIndexOf("! "), left.lastIndexOf("? "), left.lastIndexOf("\n"));
  const start = lb === -1 ? 0 : lb + 1;
  const rm = right.match(/[.!?](\s|$)|\n/);
  const end = rm ? (rm.index as number) + 1 : right.length;
  return (left.slice(start) + "___" + right.slice(0, end)).trim();
}

async function suggestOne(
  label: string,
  theme: string,
  avoid: string[] = [],
  sentence = ""
): Promise<string> {
  const avoidAll = [...avoid, ...recentSample(50)];
  const msg = await ai.messages.create({
    model: MODEL,
    max_tokens: 24,
    system: FUNNY_WORD_SYSTEM,
    messages: [
      {
        role: "user",
        content:
          `Give me the funniest "${label}".` +
          (sentence
            ? ` It will be dropped into this sentence: "${sentence}". Use the sentence ONLY to get ` +
              `the grammar right — the right part of speech, singular vs plural, verb tense — so it ` +
              `reads correctly (no double verbs). Do NOT pick a word that fits the topic of the sentence. `
            : ` `) +
          `The whole joke of Mad Libs is a word that's bizarrely OUT OF PLACE, so make it wildly ` +
          `unrelated to what the sentence${theme ? ` and the "${theme}" story` : ""} are about — ` +
          `random and absurd, not sensible. Reach into ${randomAngle()} or ${randomAngle()} for something unexpected.` +
          (avoidAll.length ? ` It must be different from: ${avoidAll.join(", ")}.` : "") +
          ` Just the word.`,
      },
    ],
  });
  return sanitizeWord(msg.content.map((b: any) => (b.type === "text" ? b.text : "")).join(""), label);
}

const CHECK_SYSTEM =
  "You are a strict but fun editor checking Mad Libs fills. A fill is GOOD if the sentence " +
  "reads as grammatically correct AND lands as funny/absurd. The word is allowed to be random " +
  "and nonsensical in meaning — that's the joke — but the sentence must NOT be grammatically " +
  "broken (e.g. two verbs in a row, wrong part of speech, a plural where a singular is needed). " +
  "Any replacement you give must be a SINGLE word (or a short 2-3 word name) that fills ONLY the " +
  "blank — NEVER a phrase, clause, or rewritten sentence. Reply with JSON only.";

// Check one filled sentence; returns a replacement word if it reads badly.
async function verifyInSentence(sentence: string, word: string, label: string) {
  const filled = sentence.replace("___", word);
  try {
    const msg = await ai.messages.create({
      model: MODEL,
      max_tokens: 60,
      system: CHECK_SYSTEM,
      messages: [
        {
          role: "user",
          content:
            `Blank type: ${label}\nFilled sentence: "${filled}"\n\n` +
            `Is it grammatically correct AND funny? If yes: {"ok":true}. ` +
            `If the grammar is broken, reply {"ok":false,"better":"<a funnier word of the SAME type that makes the sentence read correctly>"}. JSON only.`,
        },
      ],
    });
    const text = msg.content.map((b: any) => (b.type === "text" ? b.text : "")).join("");
    const v = extractJson(text);
    return { ok: v?.ok !== false, better: sanitizeWord(v?.better || "", label) };
  } catch {
    return { ok: true, better: "" }; // never block a fill on a checker hiccup
  }
}

// Batch-check every filled sentence for the Lucky button in a single call.
async function verifyBatch(
  template: string,
  blanks: { id: number; label: string }[],
  byId: Record<number, string>
): Promise<Record<number, string>> {
  const labelById: Record<number, string> = {};
  blanks.forEach((b) => { labelById[b.id] = b.label; });
  const lines = blanks.map((b) => {
    const s = sentenceFor(template, b.id).replace("___", byId[b.id] || "___");
    return `${b.id}. (${b.label}) "${s}"`;
  });
  try {
    const msg = await ai.messages.create({
      model: MODEL,
      max_tokens: 500,
      system: CHECK_SYSTEM,
      messages: [
        {
          role: "user",
          content:
            `Check each filled Mad Libs sentence below. For any whose grammar is broken (e.g. two ` +
            `verbs in a row, wrong part of speech), supply a funnier replacement of the SAME type ` +
            `that reads correctly. Words may be absurd in meaning — only fix broken grammar.\n\n` +
            lines.join("\n") +
            `\n\nReply ONLY JSON: {"fixes":[{"id":<n>,"better":"<word>"}]} listing ONLY the ids that need fixing.`,
        },
      ],
    });
    const text = msg.content.map((b: any) => (b.type === "text" ? b.text : "")).join("");
    const parsed = extractJson(text);
    const fixes: Record<number, string> = {};
    for (const f of parsed?.fixes ?? []) {
      const id = Number(f?.id);
      const better = sanitizeWord(f?.better || "", labelById[id] || "");
      if (!Number.isNaN(id) && better) fixes[id] = better;
    }
    return fixes;
  } catch {
    return {};
  }
}

// Fill every blank at once — one call for collective comedy, then repair any
// missing/duplicate slots individually so no two fills are ever identical.
async function fillAll(
  theme: string,
  blanks: { id: number; label: string }[],
  template = ""
) {
  const list = blanks
    .map((b) => {
      const s = template ? sentenceFor(template, b.id) : "";
      // Show the sentence only as a GRAMMAR guide (part of speech, singular/plural),
      // never as a topic to match — the fill should clash with it, not fit it.
      return s ? `${b.id}. ${b.label} — must fit the grammar of: "${s}"` : `${b.id}. ${b.label}`;
    })
    .join("\n");
  const recent = recentSample(70);
  let byId: Record<number, string> = {};
  try {
    const msg = await ai.messages.create({
      model: MODEL,
      max_tokens: 900,
      system: FUNNY_WORD_SYSTEM,
      messages: [
        {
          role: "user",
          content:
            `This is a Mad Libs about "${theme || "anything"}". The comedy comes from words that are ` +
            `BIZARRELY OUT OF PLACE, so do NOT pick words that fit the story — pick words that clash ` +
            `with it and surprise. A sentence is shown for each blank ONLY so you get the grammar right ` +
            `(part of speech, singular vs plural, verb tense); ignore its topic when choosing the word.\n\n` +
            `Fill each blank with the funniest, most unexpected word that fits its grammar — random and ` +
            `absurd beats sensible every time. Every word MUST be different from all the others. Avoid your ` +
            `usual go-to jokes — for inspiration, roam through territory like ${randomAngle()}, ` +
            `${randomAngle()}, and ${randomAngle()}. Family-friendly.\n\n` +
            (recent.length
              ? `Do NOT reuse any of these recently-used words: ${recent.join(", ")}.\n\n`
              : "") +
            `Blanks:\n${list}\n\n` +
            `Return ONLY JSON: {"words":[{"id":<number>,"word":"<word>"}, ...]} with an entry for every id.`,
        },
      ],
    });
    const text = msg.content.map((b: any) => (b.type === "text" ? b.text : "")).join("");
    const parsed = extractJson(text);
    const labelById: Record<number, string> = {};
    blanks.forEach((b) => { labelById[b.id] = b.label; });
    for (const w of parsed?.words ?? []) {
      const id = Number(w?.id);
      if (!Number.isNaN(id)) byId[id] = sanitizeWord(w?.word, labelById[id] || "");
    }
  } catch {
    byId = {};
  }

  // Check each fill back in its sentence; swap out any with broken grammar.
  if (template) {
    const fixes = await verifyBatch(template, blanks, byId);
    for (const id of Object.keys(fixes)) byId[Number(id)] = fixes[Number(id)];
  }

  const seen = new Set<string>();
  const out: { id: number; word: string }[] = [];
  for (const b of blanks) {
    let word = byId[b.id] || "";
    let guard = 0;
    while ((!word || seen.has(word.toLowerCase())) && guard < 3) {
      word = await suggestOne(b.label, theme, [...seen], template ? sentenceFor(template, b.id) : "");
      guard++;
    }
    if (!word || seen.has(word.toLowerCase())) word = (word || "thingamajig") + (out.length + 1);
    seen.add(word.toLowerCase());
    out.push({ id: b.id, word });
  }
  remember(out.map((o) => o.word));
  return out;
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

    // --- Suggest a single word for one blank ("think of a word for me") ---
    if (url.pathname === "/api/word" && req.method === "POST") {
      try {
        if (!process.env.SANTAI_AI_TOKEN) {
          return Response.json({ error: "AI is not configured." }, { status: 503 });
        }
        const body = await req.json().catch(() => ({}));
        const label = String(body?.label ?? "").trim().slice(0, 80);
        const theme = String(body?.theme ?? "").trim().slice(0, 200);
        const template = String(body?.template ?? "");
        const id = Number(body?.id);
        const avoid = Array.isArray(body?.avoid)
          ? body.avoid.map((w: any) => String(w).trim()).filter(Boolean).slice(0, 40)
          : [];
        if (!label) return Response.json({ error: "No label." }, { status: 400 });

        const sentence = template && !Number.isNaN(id) ? sentenceFor(template, id) : "";
        let word = await suggestOne(label, theme, avoid, sentence);
        // Check the word back in its sentence; swap it if the grammar breaks.
        if (sentence && word) {
          const v = await verifyInSentence(sentence, word, label);
          if (!v.ok && v.better) word = v.better;
        }
        if (!word) return Response.json({ error: "No word." }, { status: 502 });
        remember([word]);
        return Response.json({ word });
      } catch (err) {
        console.error("word failed:", err);
        return Response.json({ error: "Couldn't think of one." }, { status: 500 });
      }
    }

    // --- Fill every blank at once ("I'm feeling lucky") ---
    if (url.pathname === "/api/fill" && req.method === "POST") {
      try {
        if (!process.env.SANTAI_AI_TOKEN) {
          return Response.json({ error: "AI is not configured." }, { status: 503 });
        }
        const body = await req.json().catch(() => ({}));
        const theme = String(body?.theme ?? "").trim().slice(0, 200);
        const template = String(body?.template ?? "");
        const blanks = Array.isArray(body?.blanks)
          ? body.blanks
              .map((b: any) => ({ id: Number(b?.id), label: String(b?.label ?? "").slice(0, 80) }))
              .filter((b: any) => !Number.isNaN(b.id) && b.label)
              .slice(0, 20)
          : [];
        if (!blanks.length) return Response.json({ error: "No blanks." }, { status: 400 });

        const words = await fillAll(theme, blanks, template);
        return Response.json({ words });
      } catch (err) {
        console.error("fill failed:", err);
        return Response.json({ error: "Couldn't fill them in." }, { status: 500 });
      }
    }

    // --- Saved-story gallery ---
    if (url.pathname === "/api/stories" && req.method === "GET") {
      const rows = db
        .query("SELECT * FROM stories ORDER BY id DESC")
        .all() as any[];
      const stories = rows.map((r) => ({
        id: r.id,
        title: r.title,
        prompt: r.prompt,
        template: r.template,
        blanks: JSON.parse(r.blanks),
        values: JSON.parse(r.vals),
        created_at: r.created_at,
      }));
      return Response.json({ stories });
    }

    if (url.pathname === "/api/stories" && req.method === "POST") {
      try {
        const body = await req.json().catch(() => ({}));
        const title = String(body?.title ?? "").trim().slice(0, 200) || "Untitled";
        const prompt = String(body?.prompt ?? "").trim().slice(0, 300);
        const template = String(body?.template ?? "");
        const blanks = body?.blanks;
        const values = body?.values;
        if (!template || !Array.isArray(blanks) || !Array.isArray(values)) {
          return Response.json({ error: "Nothing to save." }, { status: 400 });
        }
        const created_at = new Date().toISOString();
        const info = db
          .query(
            "INSERT INTO stories (title, prompt, template, blanks, vals, created_at) VALUES (?, ?, ?, ?, ?, ?)"
          )
          .run(
            title,
            prompt,
            template,
            JSON.stringify(blanks),
            JSON.stringify(values),
            created_at
          );
        return Response.json({ id: Number(info.lastInsertRowid), created_at });
      } catch (err) {
        console.error("save failed:", err);
        return Response.json({ error: "Couldn't save story." }, { status: 500 });
      }
    }

    const delMatch = url.pathname.match(/^\/api\/stories\/(\d+)$/);
    if (delMatch && req.method === "DELETE") {
      db.query("DELETE FROM stories WHERE id = ?").run(Number(delMatch[1]));
      return Response.json({ ok: true });
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
