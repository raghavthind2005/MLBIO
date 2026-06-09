import json, re
from collections import Counter

f = '/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses_gemma4.json'
results = json.load(open(f))
valid = [r for r in results if r.get('clean_response','').strip()]

truncated = []
for r in valid:
    cr = r.get('clean_response','').strip()
    if len(cr) > 200 and cr and cr[-1] not in '.!?)"\'':
        truncated.append(r)

print('\n=== Truncation Deep-Dive: %d flagged responses ===' % len(truncated))

# Categorize by last character
last_chars = Counter(r['clean_response'].strip()[-1] for r in truncated)
print('\nLast character distribution:')
for ch, cnt in last_chars.most_common(20):
    print('  %r : %d' % (ch, cnt))

# Categorize
letter_end   = [r for r in truncated if re.match(r'^[A-E]$', r['clean_response'].strip()[-1])]
number_end   = [r for r in truncated if r['clean_response'].strip()[-1].isdigit()]
colon_end    = [r for r in truncated if r['clean_response'].strip()[-1] == ':']
bracket_end  = [r for r in truncated if r['clean_response'].strip()[-1] in ']}']
word_end     = [r for r in truncated if r['clean_response'].strip()[-1].isalpha()
                and r['clean_response'].strip()[-1] not in 'ABCDE']

print('\n--- Category breakdown ---')
print('  Ends with option letter (A-E):  %d  (multi_choice answer — OK)' % len(letter_end))
print('  Ends with digit:                %d  (numeric answer — OK)' % len(number_end))
print('  Ends with colon:                %d  (possibly mid-sentence)' % len(colon_end))
print('  Ends with bracket/brace:        %d  (likely math/code — OK)' % len(bracket_end))
print('  Ends with other letter:         %d  (check these)' % len(word_end))
print('  Other:                          %d' % (
    len(truncated) - len(letter_end) - len(number_end) - len(colon_end) - len(bracket_end) - len(word_end)))

# Show colon-ending (most likely real truncation)
if colon_end:
    print('\n--- Colon-ending responses (most suspicious) ---')
    for r in colon_end[:5]:
        cr = r['clean_response'].strip()
        print('\n  [%s] id=%s qtype=%s elapsed=%.1fs' % (
            r['subset'], r['id'], r['question_type'], r['elapsed_s']))
        print('  Last 200 chars: ...%s' % cr[-200:])

# Show word-ending (could be mid-sentence)
if word_end:
    print('\n--- Other-letter-ending responses (sample) ---')
    for r in word_end[:5]:
        cr = r['clean_response'].strip()
        print('\n  [%s] id=%s qtype=%s elapsed=%.1fs' % (
            r['subset'], r['id'], r['question_type'], r['elapsed_s']))
        print('  Last 200 chars: ...%s' % cr[-200:])

# Show letter-ending (should be fine)
if letter_end:
    print('\n--- Option-letter-ending responses (sample — should be OK) ---')
    for r in letter_end[:3]:
        cr = r['clean_response'].strip()
        print('\n  [%s] id=%s qtype=%s' % (r['subset'], r['id'], r['question_type']))
        print('  Last 100 chars: ...%s' % cr[-100:])

# Show number-ending (should be fine)
if number_end:
    print('\n--- Number-ending responses (sample — should be OK) ---')
    for r in number_end[:3]:
        cr = r['clean_response'].strip()
        print('\n  [%s] id=%s qtype=%s' % (r['subset'], r['id'], r['question_type']))
        print('  Last 100 chars: ...%s' % cr[-100:])

# Check response length distribution for flagged vs non-flagged
flag_lens = sorted([len(r['clean_response']) for r in truncated])
ok_lens   = sorted([len(r['clean_response']) for r in valid if r not in truncated])
def pct(lst, p): return lst[int(len(lst)*p//100)] if lst else 0
print('\n--- Response length (chars) ---')
print('  Flagged:     p50=%d  p90=%d  p99=%d  max=%d' % (
    pct(flag_lens,50), pct(flag_lens,90), pct(flag_lens,99), flag_lens[-1] if flag_lens else 0))
print('  Not flagged: p50=%d  p90=%d  p99=%d  max=%d' % (
    pct(ok_lens,50), pct(ok_lens,90), pct(ok_lens,99), ok_lens[-1] if ok_lens else 0))

# Final verdict
truly_suspect = colon_end + word_end
print('\n=== VERDICT ===')
print('  Likely false positives (letter/number/bracket endings): %d' % (
    len(letter_end) + len(number_end) + len(bracket_end)))
print('  Genuinely suspect (colon or mid-word endings):          %d' % len(truly_suspect))
print()
