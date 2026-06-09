import json, re
from collections import defaultdict

f = '/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses_gemma4.json'
results = json.load(open(f))
valid = [r for r in results if r.get('clean_response','').strip()]
print('\n' + '='*60)
print('  Quality Report: %d/%d valid entries' % (len(valid), len(results)))
print('='*60)

think = [r for r in valid if r.get('has_thinking')]
no_think = [r for r in valid if not r.get('has_thinking')]
words = sorted([r['thinking_words'] for r in think])
print('\n[Thinking]')
print('  ON:  %d/%d (%d%%)' % (len(think), len(valid), 100*len(think)//max(1,len(valid))))
print('  OFF: %d samples' % len(no_think))
if words:
    def p(pct): return words[int(len(words)*pct//100)]
    print('  Words  min:%d  p10:%d  p50:%d  p90:%d  p99:%d  max:%d' % (
        words[0], p(10), p(50), p(90), p(99), words[-1]))

truncated_think = [r for r in valid if r.get('thinking_words',0) > 10000]
truncated_resp = []
for r in valid:
    cr = r.get('clean_response','').strip()
    if len(cr) > 200 and cr and cr[-1] not in '.!?)"\'':
        truncated_resp.append(r)
short_resp = [r for r in valid if len(r.get('clean_response','').strip()) < 5]
print('\n[Truncation / Quality]')
print('  Thinking possibly truncated (>10k words): %d' % len(truncated_think))
print('  Response possibly truncated (no end punct): %d' % len(truncated_resp))
print('  Very short responses (<5 chars): %d' % len(short_resp))
if short_resp:
    for r in short_resp[:3]:
        print('    id=%s subset=%s: %r' % (r['id'], r['subset'], r['clean_response'][:80]))

times = sorted([r['elapsed_s'] for r in valid])
def pt(pct): return times[int(len(times)*pct//100)]
slow = [r for r in valid if r['elapsed_s'] > 150]
print('\n[Timing (s/sample)]')
print('  min:%.1f  p50:%.1f  p90:%.1f  p99:%.1f  max:%.1f' % (
    times[0], pt(50), pt(90), pt(99), times[-1]))
print('  Slow samples (>150s): %d' % len(slow))
if slow:
    for r in slow[:3]:
        print('    id=%s subset=%s elapsed=%.1fs think_words=%d' % (
            r['id'], r['subset'], r['elapsed_s'], r.get('thinking_words',0)))

mc = [r for r in valid if r.get('question_type') == 'multi_choice']
extracted = [r for r in mc if r.get('pred_letter')]
print('\n[Answer Extraction]')
print('  multi_choice: %d samples' % len(mc))
print('  pred_letter extracted: %d/%d (%d%%)' % (
    len(extracted), len(mc), 100*len(extracted)//max(1,len(mc))))

print('\n[Per-subset]')
for subset in ['halu', 'reason']:
    sub = [r for r in valid if r['subset'] == subset]
    sub_think = sum(1 for r in sub if r.get('has_thinking'))
    qtypes = defaultdict(int)
    for r in sub: qtypes[r['question_type']] += 1
    print('  %s: %d samples | thinking %d/%d | %s' % (
        subset, len(sub), sub_think, len(sub), dict(qtypes)))

print('\n[Sample Outputs]')
for subset in ['halu', 'reason']:
    sub = [r for r in valid if r['subset'] == subset]
    for r in sub[:2]:
        print('\n  [%s] id=%s qtype=%s elapsed=%.1fs' % (
            subset, r['id'], r['question_type'], r['elapsed_s']))
        print('  Q: %s' % r['question'][:100])
        print('  GT: %s' % str(r['gt_answer'])[:80])
        if r.get('has_thinking'):
            tp = r['raw_response'].split('</think>')[0].replace('<think>','').strip()
            print('  <think>: %s...' % tp[:150])
        print('  Answer: %s' % r['clean_response'][:150])

print('\n' + '='*60 + '\n')
