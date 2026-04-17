const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const multer = require("multer");
const dotenv = require("dotenv");
const path = require("path");
const pdfParse = require("pdf-parse");
const { createWorker } = require("tesseract.js");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const openRouterKey = process.env.OPENROUTER_API_KEY;
const openRouterModel = process.env.OPENROUTER_MODEL || "openai/gpt-4o-mini";
const corsOrigin = process.env.CORS_ORIGIN || "*";

app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: corsOrigin === "*" ? true : corsOrigin }));
app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname)));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 }
});

async function callOpenRouter(messages) {
  if (!openRouterKey) {
    throw new Error("OPENROUTER_API_KEY missing in environment.");
  }

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${openRouterKey}`,
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "CareBridge Backend"
    },
    body: JSON.stringify({
      model: openRouterModel,
      messages
    })
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenRouter request failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  return data?.choices?.[0]?.message?.content?.trim() || "No AI response generated.";
}

async function extractTextWithOCR(fileBuffer) {
  const worker = await createWorker("eng");
  try {
    const {
      data: { text }
    } = await worker.recognize(fileBuffer);
    return (text || "").trim();
  } finally {
    await worker.terminate();
  }
}

async function extractPrescriptionText(file) {
  const isPdf = file.mimetype === "application/pdf" || file.originalname.toLowerCase().endsWith(".pdf");
  if (isPdf) {
    try {
      const parsed = await pdfParse(file.buffer);
      const text = (parsed?.text || "").trim();
      if (text.length > 40) return text;
    } catch (err) {
      console.warn("PDF text extraction failed, fallback OCR:", err.message);
    }
  }
  return extractTextWithOCR(file.buffer);
}

app.get("/api/health", (req, res) => {
  res.json({ ok: true, service: "carebridge-backend", model: openRouterModel });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message, systemPrompt } = req.body || {};
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message is required" });
    }

    const reply = await callOpenRouter([
      {
        role: "system",
        content:
          systemPrompt ||
          "You are CareBridge AI assistant. Give safe and concise healthcare guidance. For emergencies ask user to call emergency services."
      },
      { role: "user", content: message }
    ]);

    return res.json({ reply });
  } catch (error) {
    console.error("/api/chat error:", error);
    return res.status(500).json({ error: "AI chat failed", details: error.message });
  }
});

app.post("/api/analyze-prescription", upload.single("prescription"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "prescription file is required" });
    }

    const extractedText = await extractPrescriptionText(req.file);
    const analysis = await callOpenRouter([
      {
        role: "system",
        content:
          "You are a medical prescription explainer. Use simple language. Output: medicines, when/how to take, what to do, what to avoid, food guidance, warning signs. Mention uncertainty if handwriting/text is unclear."
      },
      {
        role: "user",
        content: `Analyze this prescription text and explain for patient:\n\n${extractedText || "No readable text found."}`
      }
    ]);

    return res.json({ analysis, extractedText });
  } catch (error) {
    console.error("/api/analyze-prescription error:", error);
    return res.status(500).json({ error: "Prescription analysis failed", details: error.message });
  }
});

app.post("/api/analyze-symptoms", async (req, res) => {
  try {
    const { symptoms } = req.body || {};
    if (!symptoms || typeof symptoms !== "string") {
      return res.status(400).json({ error: "symptoms is required" });
    }

    const analysis = await callOpenRouter([
      {
        role: "system",
        content:
          "You are a medical triage assistant. Use simple language and output headings: likely causes, what to do now, what to avoid, food guidance, red flags. This is not final diagnosis."
      },
      { role: "user", content: symptoms }
    ]);

    return res.json({ analysis });
  } catch (error) {
    console.error("/api/analyze-symptoms error:", error);
    return res.status(500).json({ error: "Symptom analysis failed", details: error.message });
  }
});

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

function startServer(preferredPort) {
  const server = app.listen(preferredPort, () => {
    console.log(`CareBridge backend running on http://localhost:${preferredPort}`);
  });

  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      const nextPort = Number(preferredPort) + 1;
      console.warn(`Port ${preferredPort} is busy. Retrying on ${nextPort}...`);
      startServer(nextPort);
      return;
    }
    throw error;
  });
}

startServer(Number(port));
