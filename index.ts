/**
 * Guardian extension for pi - LLM auto-approval of risky tool calls.
 *
 * Port of the OpenAI Codex "guardian" auto-review design (Apache-2.0,
 * github.com/openai/codex, codex-rs/core/src/guardian/) onto pi's extension
 * API. Layering mirrors Codex:
 *
 *   1. Static gates: read-only tools and an allowlist of safe bash commands
 *      run without review; writes/edits inside the workspace run without
 *      review (pi has no sandbox, so this stands in for workspace-write).
 *   2. Everything else goes to a guardian model that judges the exact action
 *      against a policy (risk_level x user_authorization -> allow/deny),
 *      using a compact transcript as untrusted evidence.
 *   3. Fail closed: timeout, parse failure, or missing model never silently
 *      allows. With a UI the human is prompted; headless, the action blocks.
 *   4. Circuit breaker: repeated denials disable auto-review and fall back
 *      to manual prompts for the rest of the session.
 */

import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, resolve, sep } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Configuration constants (mirroring codex-rs/core/src/guardian/mod.rs; token
// limits converted to chars at ~4 chars/token)
// ---------------------------------------------------------------------------

const GUARDIAN_PROVIDER = "anthropic";
const GUARDIAN_MODEL_ID = "claude-opus-5";
const GUARDIAN_REVIEW_TIMEOUT_MS = 90_000;
const GUARDIAN_MAX_ATTEMPTS = 3;

const MAX_CONSECUTIVE_DENIALS_PER_TURN = 3;
const DENIAL_WINDOW_SIZE = 50;
const MAX_WINDOW_DENIALS = 10;

const MAX_TRANSCRIPT_ENTRIES = 40;
const MAX_CHARS_PER_MESSAGE = 8_000;
const MAX_CHARS_PER_TOOL_ENTRY = 4_000;
const MAX_TRANSCRIPT_CHARS = 80_000;
const MAX_ACTION_CHARS = 64_000;

const READ_ONLY_TOOLS = new Set(["read", "grep", "find", "ls"]);
const WORKSPACE_WRITE_TOOLS = new Set(["write", "edit"]);

/** First tokens of bash segments that never need review. */
const SAFE_BASH_COMMANDS = new Set([
	"ls", "cat", "head", "tail", "wc", "pwd", "echo", "true", "which",
	"whoami", "date", "env", "printenv", "file", "stat", "du", "df",
	"grep", "rg", "find", "cd", "git",
]);
const SAFE_GIT_SUBCOMMANDS = new Set(["status", "diff", "log", "show", "branch", "remote"]);
const UNSAFE_FIND_FLAGS = /-(delete|exec|execdir|ok|okdir)\b/;
/** Redirection / substitution makes a segment-wise safety check unsound. */
const UNSAFE_SHELL_SYNTAX = /[<>`]|\$\(/;

// Project override -> user override -> bundled Codex default policy.
const PROJECT_POLICY_PATH = ".pi/guardian-policy.md";
const USER_POLICY_PATH = join(homedir(), ".pi", "agent", "guardian-policy.md");

/** Set PI_GUARDIAN_LOG=/path/to/file to append one JSON line per review. */
const GUARDIAN_LOG_PATH = process.env.PI_GUARDIAN_LOG;

const OUTPUT_CONTRACT = `
# Output Contract
You cannot run tools in this environment. Judge only from the evidence provided; when an investigation would be required to resolve critical uncertainty, lean conservative and explain the uncertainty in your rationale.
Your entire reply must be a single strict JSON object of the form:
{"risk_level": "low|medium|high|critical", "user_authorization": "unknown|low|medium|high", "outcome": "allow|deny", "rationale": "one concise sentence"}
Only "outcome" is required; for clearly low-risk actions you may reply {"outcome": "allow"}.
Do not wrap the JSON in markdown fences or add any other text.`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type RiskLevel = "low" | "medium" | "high" | "critical";
type UserAuthorization = "unknown" | "low" | "medium" | "high";

interface GuardianAssessment {
	outcome: "allow" | "deny";
	risk_level?: RiskLevel;
	user_authorization?: UserAuthorization;
	rationale?: string;
}

interface GuardianStats {
	reviews: number;
	allowed: number;
	denied: number;
	overridden: number;
	failures: number;
}

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

const extensionDir = new URL(".", import.meta.url).pathname;

function loadPolicyTemplate(): string {
	return readFileSync(join(extensionDir, "prompts", "policy_template.md"), "utf8");
}

function loadTenantPolicy(): string {
	const projectPolicy = resolve(process.cwd(), PROJECT_POLICY_PATH);
	if (existsSync(projectPolicy)) return readFileSync(projectPolicy, "utf8");
	if (existsSync(USER_POLICY_PATH)) return readFileSync(USER_POLICY_PATH, "utf8");
	return readFileSync(join(extensionDir, "prompts", "policy.md"), "utf8");
}

function truncate(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	return `${text.slice(0, maxChars)}\n<guardian_truncated original_chars="${text.length}"/>`;
}

interface SessionContentBlock {
	type?: string;
	text?: string;
	name?: string;
	arguments?: Record<string, unknown>;
}

function contentBlocks(content: unknown): SessionContentBlock[] {
	if (typeof content === "string") return [{ type: "text", text: content }];
	if (!Array.isArray(content)) return [];
	return content.filter((b): b is SessionContentBlock => !!b && typeof b === "object");
}

/** Compact transcript: most recent entries, per-entry and total char caps. */
function buildTranscript(ctx: ExtensionContext): string {
	const entries = ctx.sessionManager
		.getBranch()
		.filter((e: { type: string }) => e.type === "message")
		.slice(-MAX_TRANSCRIPT_ENTRIES);

	const sections: string[] = [];
	for (const entry of entries as Array<{
		message?: { role?: string; toolName?: string; content?: unknown; isError?: boolean };
	}>) {
		const message = entry.message;
		if (!message?.role) continue;
		const blocks = contentBlocks(message.content);

		if (message.role === "user" || message.role === "assistant") {
			const text = blocks
				.filter((b) => b.type === "text" && typeof b.text === "string")
				.map((b) => b.text as string)
				.join("\n")
				.trim();
			if (text) {
				const label = message.role === "user" ? "User" : "Assistant";
				sections.push(`${label}: ${truncate(text, MAX_CHARS_PER_MESSAGE)}`);
			}
			if (message.role === "assistant") {
				for (const b of blocks) {
					if (b.type === "toolCall" && typeof b.name === "string") {
						const args = JSON.stringify(b.arguments ?? {});
						sections.push(`Assistant called tool ${b.name} with ${truncate(args, MAX_CHARS_PER_TOOL_ENTRY)}`);
					}
				}
			}
		} else if (message.role === "toolResult") {
			const text = blocks
				.filter((b) => b.type === "text" && typeof b.text === "string")
				.map((b) => b.text as string)
				.join("\n")
				.trim();
			const errorTag = message.isError ? " (error)" : "";
			sections.push(
				`Tool result${errorTag} from ${message.toolName ?? "unknown"}: ${truncate(text, MAX_CHARS_PER_TOOL_ENTRY)}`,
			);
		}
	}

	let transcript = sections.join("\n\n");
	if (transcript.length > MAX_TRANSCRIPT_CHARS) {
		transcript = `<guardian_truncated dropped_leading_chars="${transcript.length - MAX_TRANSCRIPT_CHARS}"/>\n${transcript.slice(-MAX_TRANSCRIPT_CHARS)}`;
	}
	return transcript;
}

function buildReviewPrompt(ctx: ExtensionContext, toolName: string, input: unknown): string {
	const template = loadPolicyTemplate().replace("{{ tenant_policy_config }}", loadTenantPolicy().trim());
	const action = truncate(
		`Tool: ${toolName}\nWorking directory: ${process.cwd()}\nInput: ${JSON.stringify(input, null, 2)}`,
		MAX_ACTION_CHARS,
	);
	return [
		template.trim(),
		OUTPUT_CONTRACT.trim(),
		"# Transcript (untrusted evidence)",
		`<transcript>\n${buildTranscript(ctx)}\n</transcript>`,
		"# Planned Action (untrusted evidence)",
		`<planned_action>\n${action}\n</planned_action>`,
	].join("\n\n");
}

// ---------------------------------------------------------------------------
// Static gates
// ---------------------------------------------------------------------------

export function isSafeBashCommand(command: string): boolean {
	if (UNSAFE_SHELL_SYNTAX.test(command)) return false;
	const segments = command
		.split(/\n|;|\|\||&&|\|/)
		.map((s) => s.trim())
		.filter((s) => s.length > 0);
	if (segments.length === 0) return false;
	for (const segment of segments) {
		const words = segment.split(/\s+/);
		const head = words[0];
		if (!head || !SAFE_BASH_COMMANDS.has(head)) return false;
		if (head === "git" && !SAFE_GIT_SUBCOMMANDS.has(words[1] ?? "")) return false;
		if (head === "find" && UNSAFE_FIND_FLAGS.test(segment)) return false;
	}
	return true;
}

function isWorkspacePath(path: unknown): boolean {
	if (typeof path !== "string" || path.length === 0) return false;
	const cwd = process.cwd();
	const absolute = isAbsolute(path) ? resolve(path) : resolve(cwd, path);
	return absolute === cwd || absolute.startsWith(cwd + sep);
}

/** True when the action can run without guardian review. */
export function passesStaticGates(toolName: string, input: Record<string, unknown>): boolean {
	if (READ_ONLY_TOOLS.has(toolName)) return true;
	if (WORKSPACE_WRITE_TOOLS.has(toolName)) return isWorkspacePath(input.path);
	if (toolName === "bash" && typeof input.command === "string") {
		return isSafeBashCommand(input.command);
	}
	return false;
}

// ---------------------------------------------------------------------------
// Guardian review
// ---------------------------------------------------------------------------

export function parseVerdict(text: string): GuardianAssessment | undefined {
	const candidates = [text.trim()];
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start >= 0 && end > start) candidates.push(text.slice(start, end + 1));
	for (const candidate of candidates) {
		try {
			const parsed = JSON.parse(candidate) as Record<string, unknown>;
			if (parsed.outcome === "allow" || parsed.outcome === "deny") {
				return parsed as unknown as GuardianAssessment;
			}
		} catch {
			// try next candidate
		}
	}
	return undefined;
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
	return new Promise<T>((resolvePromise, rejectPromise) => {
		const timer = setTimeout(() => rejectPromise(new Error(`guardian review timed out after ${ms}ms`)), ms);
		promise.then(
			(value) => {
				clearTimeout(timer);
				resolvePromise(value);
			},
			(error) => {
				clearTimeout(timer);
				rejectPromise(error);
			},
		);
	});
}

export default function guardianExtension(pi: ExtensionAPI) {
	const guardianSessionId = randomUUID();
	const stats: GuardianStats = { reviews: 0, allowed: 0, denied: 0, overridden: 0, failures: 0 };

	let enabled = true;
	let breakerTripped = false;
	let consecutiveDenials = 0;
	const denialWindow: boolean[] = [];

	function recordReview(denied: boolean) {
		denialWindow.push(denied);
		if (denialWindow.length > DENIAL_WINDOW_SIZE) denialWindow.shift();
		consecutiveDenials = denied ? consecutiveDenials + 1 : 0;
		const windowDenials = denialWindow.filter(Boolean).length;
		if (consecutiveDenials >= MAX_CONSECUTIVE_DENIALS_PER_TURN || windowDenials >= MAX_WINDOW_DENIALS) {
			breakerTripped = true;
		}
	}

	function setStatus(ctx: ExtensionContext, text: string) {
		if (ctx.hasUI) ctx.ui.setStatus("guardian", text);
	}

	function logReview(toolName: string, entry: Record<string, unknown>) {
		if (!GUARDIAN_LOG_PATH) return;
		try {
			appendFileSync(GUARDIAN_LOG_PATH, `${JSON.stringify({ time: new Date().toISOString(), tool: toolName, ...entry })}\n`);
		} catch {
			// logging must never break the approval flow
		}
	}

	function resolveGuardianModel(ctx: ExtensionContext) {
		const preferred = ctx.modelRegistry.find(GUARDIAN_PROVIDER, GUARDIAN_MODEL_ID);
		if (preferred && ctx.modelRegistry.hasConfiguredAuth(preferred)) return preferred;
		// Codex parity: fall back to the session's main model when the preferred
		// review model is unavailable.
		if (ctx.model && ctx.modelRegistry.hasConfiguredAuth(ctx.model)) return ctx.model;
		return undefined;
	}

	async function requestVerdict(
		ctx: ExtensionContext,
		toolName: string,
		input: unknown,
	): Promise<GuardianAssessment> {
		const model = resolveGuardianModel(ctx);
		if (!model) throw new Error("no guardian model with configured auth");
		const prompt = buildReviewPrompt(ctx, toolName, input);
		const messages = [
			{
				role: "user" as const,
				content: [{ type: "text" as const, text: prompt }],
				timestamp: Date.now(),
			},
		];

		let lastError: unknown;
		for (let attempt = 1; attempt <= GUARDIAN_MAX_ATTEMPTS; attempt++) {
			try {
				const response = await withTimeout(
					ctx.modelRegistry.complete(model, { messages }, { effort: "low", sessionId: guardianSessionId }),
					GUARDIAN_REVIEW_TIMEOUT_MS,
				);
				const text = response.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");
				const verdict = parseVerdict(text);
				if (verdict) return verdict;
				lastError = new Error(`unparseable guardian verdict: ${text.slice(0, 200)}`);
			} catch (error) {
				lastError = error;
			}
		}
		throw lastError instanceof Error ? lastError : new Error(String(lastError));
	}

	/** Manual fallback: prompt the user when the guardian can't decide. */
	async function askUser(ctx: ExtensionContext, title: string, detail: string): Promise<boolean> {
		if (!ctx.hasUI) return false;
		return await ctx.ui.confirm(title, detail);
	}

	function denialReason(toolName: string, verdict: GuardianAssessment): string {
		const risk = verdict.risk_level ?? "unknown";
		const auth = verdict.user_authorization ?? "unknown";
		const rationale = verdict.rationale ?? "no rationale provided";
		// Post-denial agent instructions mirror codex guardian/review.rs.
		return (
			`Automatic approval review denied ${toolName} (risk: ${risk}, authorization: ${auth}): ${rationale} ` +
			`Do not attempt to work around this denial. Proceed only with a materially safer alternative, ` +
			`or ask the user to explicitly approve this exact action.`
		);
	}

	pi.on("session_start", async (_event, ctx) => {
		setStatus(ctx, "guardian: auto");
	});

	pi.on("before_agent_start", async () => {
		// New user prompt = new turn: reset the consecutive-denial counter.
		consecutiveDenials = 0;
	});

	pi.on("tool_call", async (event, ctx) => {
		if (!enabled) return undefined;
		const input = event.input as Record<string, unknown>;
		if (passesStaticGates(event.toolName, input)) return undefined;

		if (breakerTripped) {
			const approved = await askUser(
				ctx,
				"Guardian paused (circuit breaker)",
				`Run ${event.toolName}?\n\n${truncate(JSON.stringify(input, null, 2), 2_000)}`,
			);
			if (approved) return undefined;
			return { block: true, reason: "Guardian circuit breaker active; user did not approve the action." };
		}

		setStatus(ctx, "guardian: reviewing…");
		stats.reviews += 1;
		let verdict: GuardianAssessment;
		try {
			verdict = await requestVerdict(ctx, event.toolName, input);
		} catch (error) {
			stats.failures += 1;
			setStatus(ctx, "guardian: auto");
			// Fail closed: never silently allow on guardian failure.
			const message = error instanceof Error ? error.message : String(error);
			logReview(event.toolName, { result: "failure", error: message });
			const approved = await askUser(
				ctx,
				"Guardian review failed",
				`${message}\n\nRun ${event.toolName} anyway?\n\n${truncate(JSON.stringify(input, null, 2), 2_000)}`,
			);
			if (approved) return undefined;
			return { block: true, reason: `Guardian review failed (${message}); action blocked (fail closed).` };
		}
		setStatus(ctx, "guardian: auto");
		logReview(event.toolName, {
			result: verdict.outcome,
			risk: verdict.risk_level,
			authorization: verdict.user_authorization,
			rationale: verdict.rationale,
		});

		if (verdict.outcome === "allow") {
			stats.allowed += 1;
			recordReview(false);
			return undefined;
		}

		stats.denied += 1;
		recordReview(true);
		if (breakerTripped && ctx.hasUI) {
			ctx.ui.notify("Guardian circuit breaker tripped; falling back to manual prompts.", "warning");
			setStatus(ctx, "guardian: paused");
		}

		const reason = denialReason(event.toolName, verdict);
		const approved = await askUser(
			ctx,
			"Guardian denied this action",
			`${verdict.rationale ?? "No rationale."}\n\nrisk: ${verdict.risk_level ?? "?"} | authorization: ${verdict.user_authorization ?? "?"}\n\nAllow anyway?`,
		);
		if (approved) {
			stats.overridden += 1;
			// Manual approval mirrors Codex post-denial approval: trust the user.
			consecutiveDenials = 0;
			return undefined;
		}
		return { block: true, reason };
	});

	pi.registerCommand("guardian", {
		description: "Toggle the guardian or show its stats (usage: /guardian [on|off|stats])",
		handler: async (args, ctx) => {
			const arg = (args ?? "").trim();
			if (arg === "on") {
				enabled = true;
				breakerTripped = false;
				consecutiveDenials = 0;
				denialWindow.length = 0;
				setStatus(ctx, "guardian: auto");
				ctx.ui.notify("Guardian enabled", "info");
				return;
			}
			if (arg === "off") {
				enabled = false;
				setStatus(ctx, "guardian: off");
				ctx.ui.notify("Guardian disabled", "warning");
				return;
			}
			const state = !enabled ? "off" : breakerTripped ? "paused (circuit breaker)" : "auto";
			ctx.ui.notify(
				`Guardian ${state} - reviews: ${stats.reviews}, allowed: ${stats.allowed}, denied: ${stats.denied}, overridden: ${stats.overridden}, failures: ${stats.failures}`,
				"info",
			);
		},
	});
}
