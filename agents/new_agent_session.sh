#!/usr/bin/env bash
# Start the supplied saved OpenAI Agent and print its raw streamed events.
# This implementation deliberately uses cURL rather than an SDK.
set -euo pipefail

API_BASE="https://api.openai.com/v1/agents"
SAVED_AGENT_ID="agent_fdf14800ea9547f1bdc8e8a1ee3e257b35b6b092975a48a3a1"
INITIAL_MESSAGE="Create a complete, practical heavy-metal requiem LP documentation package. Include an initial 10-song tracklist and the recording-plan documents needed to begin the demo phase."
DRY_RUN=false
OVERRIDE_INSTRUCTIONS="$(cat <<'EOF'
Generate comprehensive documentation for a full-length heavy metal requiem LP, including demo materials and recording session documents. Capture all relevant details necessary for planning, recording, and producing the album.

- Ensure the documentation covers:
  - Album concept and theme (requiem in heavy metal context)
  - Tracklist with song titles and brief descriptions
  - Demo recordings: session dates, performing personnel, demo notes
  - Recording plans: studio selection, session schedules, producer/engineer notes
  - Instrumentation details, gear lists, tuning specifics, and arrangement notes
  - Lyrics (full or excerpts), vocal style direction, and reference influences
  - Technical goals: sound references, production techniques, mix/mastering intentions
  - Any additional creative direction, artwork notes, or release strategy
- For reasoning: First explain the logic and thought process behind decisions on album concept, song structure, production choices, and session planning before listing the conclusions and finished documentation.
- Ensure any conclusions or final album documents appear after the preceding reasoning and planning process.
- Use clear sections for each document type.
- Output format: Present the documentation as an organized markdown file, with distinct headings for each section, lists, and tables if helpful. Use placeholders [ ] for demo audio or artwork if audio/visual is referenced.
- Examples are encouraged: Provide at least one sample album concept and at least two example song entries, using placeholders for more complex content (e.g., "[Demo Audio Placeholder]"). Extend examples to cover both the planning and finished documentation.

---

**Example Input:**
Task: Generate album plan and recording documents for a fictional heavy metal requiem LP.

**Example Output:**
## Reasoning and Planning
- The theme of a requiem aligns with heavy metal's dramatic and somber tones. I chose [Band Name] to create a concept-driven LP that blends classical requiem elements with modern metal instrumentation. Key inspirations include Metallica’s S&M and Septicflesh.
- Structurally, the album will be divided into movements mirroring a traditional requiem (Introit, Kyrie, etc.), each interpreted in a metal style. Session plans are designed for layered recording to achieve a full, orchestral sound.
- For the demo phase, early versions will focus on capturing song structure and riff ideas before developing full lyrics and arrangements.

## Album Demo and Recording Documents

### Album Concept and Theme
**Title:** Requiem for the Warriors
**Artist:** [Band Name]
**Concept:** A heavy metal requiem fusing classical forms and themes of mortality with modern aggression and orchestration.

### Tracklist
| # | Title | Description |
|:-:|-------------------|------------------------------------------------|
| 1 | Introit | Atmospheric intro, orchestral with guitars |
| 2 | Kyrie | Melodic, chant-based with powerful drums |
| 3 | Dies Irae | Fast, rhythmic, riff-driven lament |
| 4 | Sanctus | Uplifting chorus, choral elements added |
| ...| ... | ... |

### Demo Session Details
**Session 1:**
- Date: [MM/DD/YYYY]
- Personnel: [Names/Roles]
- Notes: Tracked rhythm section for “Introit” and “Dies Irae”; basic structure outlined; placeholder demo vocals used.
- Demo files: [Demo Audio Placeholder]

### Recording Plan
- Studio: [Studio Name/Location]
- Schedule: [Timeline Overview/Table]
- Producer/Engineer: [Names]
- Instrumentation:
  - Guitars: 7-string, drop C tuning
  - Bass: 5-string, overdrive used
  - Drums: Double kick emphasis
  - Orchestration: Symphonic samples layered in

### Lyrics (Excerpt)
**Dies Irae:**
“Thunder roars on judgment day,
Ashes whisper what remains*...*”

### Technical Direction
- Mix reference: Metallica S&M, Septicflesh “Communion”
- Production target: Wide, epic stereo image, layered choral vocals
- Mastering intention: Loud but dynamic

---

(For real-world documentation, more tracks, fuller lyrics, and longer planning notes should be included as appropriate.)

---

**Important Reminder:**
First reason through concept, structure, and production decisions, then provide the finished documentation in detailed markdown. Use placeholders as needed for complex assets. Provide distinct sections for each element of the documentation.
EOF
)"

require() { command -v "$1" >/dev/null || { echo "Missing required command: $1" >&2; exit 2; }; }

usage() {
  cat <<'EOF'
Usage: bash agents/new_agent_session.sh [--message TEXT] [--dry-run]

  --message TEXT  Replace the initial user message for this new session.
  --dry-run       Print the create-session JSON without calling the API.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --message)
      [[ $# -ge 2 ]] || { echo "--message requires text." >&2; exit 2; }
      INITIAL_MESSAGE="$2"
      shift 2
      ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require curl
require jq

api() {
  curl --silent --show-error --fail-with-body "$@" \
    -H "OpenAI-Beta: agents=v1" \
    -H "Authorization: Bearer $OPENAI_API_KEY"
}

# The server calls this local, side-effect-free handler only after it explicitly
# exposes a pending required action. Its JSON result is returned to the session.
tracklist_result() {
  local arguments="$1"
  jq -cn --argjson args "$arguments" '
    def songs: ["Introit: Ashes at Dawn", "Kyrie of Iron", "Dies Irae Engine",
      "Lacrimosa in C Minor", "Sanctus of the Fallen", "Agnus Dei: Steel Mercy",
      "Offertory for the Nameless", "Nocturne of Black Glass", "Requiem Aeternam",
      "Benediction: Last Light"];
    {include_plans: ($args.include_plans // false),
     pro_tools: (($args.pro_tools_folders // []) | map({folder: ., tracklist: songs})),
     documents: (($args.documents_folders // []) | map({folder: ., planning: ["scope", "personnel", "schedule", "gear", "approval"]}))}'
}

handle_required_actions() {
  local session_id="$1" session action turn_id call_id arguments output payload
  session="$(api "$API_BASE/sessions/$session_id")"
  while IFS= read -r action; do
    [[ -z "$action" ]] && continue
    turn_id="$(jq -r '.turn_id' <<<"$action")"
    call_id="$(jq -r '.call_id' <<<"$action")"
    arguments="$(jq -c '.arguments' <<<"$action")"
    if [[ "$(jq -r '.name' <<<"$action")" == "generate_tracklist_for_folders" ]]; then
      output="$(tracklist_result "$arguments")"
      payload="$(jq -n --arg turn_id "$turn_id" --arg call_id "$call_id" --arg output "$output" \
        '{events: [{type: "agent.session.input.tool_result", turn_id: $turn_id, call_id: $call_id, success: true, output: $output}]}')"
    else
      payload="$(jq -n --arg turn_id "$turn_id" --arg call_id "$call_id" \
        '{events: [{type: "agent.session.input.tool_result", turn_id: $turn_id, call_id: $call_id, success: false, error: "Unsupported local function."}]}')"
    fi
    api -X POST "$API_BASE/sessions/$session_id/events" -H "Content-Type: application/json" -d "$payload" >/dev/null
  done < <(jq -c '.required_actions[]? | select(.type == "function_call")' <<<"$session")
}

payload="$(jq -n --arg agent_id "$SAVED_AGENT_ID" --arg input "$INITIAL_MESSAGE" --arg instructions "$OVERRIDE_INSTRUCTIONS" '
  {agent_id: $agent_id,
   agent: {
     model: "gpt-6-sol",
     instructions: $instructions,
     reasoning: {effort: "medium", summary: "auto"},
     text: {format: {type: "text"}, verbosity: "medium"}
   },
   environment: {type: "none"}, input: $input, stream: true}')"

if [[ "$DRY_RUN" == true ]]; then
  printf '%s\n' "$payload"
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY to an API key for proj_GCS5STipQ3WPoOb0T1jXEmMr." >&2
  exit 2
fi

event_log="$(mktemp)"
trap 'rm -f "$event_log"' EXIT
echo "Starting saved agent session; raw server-sent events follow:" >&2
api --no-buffer -X POST "$API_BASE/sessions" -H "Content-Type: application/json" -d "$payload" | tee "$event_log"
session_id="$(sed -n 's/^data: //p' "$event_log" | jq -r 'select(.session.id?) | .session.id' | head -n1)"
[[ -n "$session_id" && "$session_id" != "null" ]] || { echo "No session ID found in event stream." >&2; exit 1; }
echo "Session: $session_id" >&2

# A function call pauses the turn. Submit every pending result, then re-open the
# server event stream so follow-up output, errors, and completion are visible.
for _ in 1 2 3 4 5; do
  session="$(api "$API_BASE/sessions/$session_id")"
  pending="$(jq '[.required_actions[]? | select(.type == "function_call")] | length' <<<"$session")"
  (( pending == 0 )) && break
  handle_required_actions "$session_id"
  api --no-buffer "$API_BASE/sessions/$session_id/events?stream=true" -H "Accept: text/event-stream"
done
