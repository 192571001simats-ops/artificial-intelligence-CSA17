# Task B4 - retrieval-based operator assistant for the intrusion detector (TF-IDF + cosine similarity,
# confidence-threshold fallback for out-of-scope requests) and a simulated speech-recognition noise channel.
import time, random, numpy as np

INTENTS = {
    'open_alerts': ['which alerts are open', 'show me the open alerts', 'list unresolved alerts', 'any active alerts right now',
                    'what alerts are still pending', 'give me all open incidents', 'are there alerts nobody has handled',
                    'display current alerts', 'open alerts please', 'what is currently flagged', 'outstanding alerts', 'alerts waiting for review', 'tickets that are not closed', 'whats open on the board', 'unacknowledged alarms', 'show me what still needs attention'],
    'explain_alert': ['why was this flagged', 'explain alert 3', 'why did the model raise this alert', 'what made connection 12 suspicious',
                      'reason for this detection', 'why is host 2041 marked as an attack', 'explain why this was classified malicious',
                      'what features triggered the alarm', 'justify this alert', 'why do you think this is an intrusion', 'on what basis was it flagged', 'which signals made it look malicious', 'break down the decision for this alert', 'what evidence supports this alarm', 'how did the model decide this', 'tell me what looked wrong in that traffic'],
    'alert_count': ['how many alerts today', 'number of alerts in the last hour', 'count the alerts', 'how many detections so far',
                    'total alerts this shift', 'how many attacks were detected', 'alert count please', 'how many incidents since morning',
                    'tell me the number of alarms', 'how busy has the detector been', 'alarm tally', 'how many have fired', 'volume of alerts since midnight', 'how many times did it alarm', 'give me the alert numbers', 'alerts raised in total'],
    'top_hosts': ['which hosts are the noisiest', 'top ten suspicious hosts', 'which machines generate most alerts', 'worst offending hosts',
                  'show hosts with the highest alert rate', 'who is attacking us the most', 'list the most flagged sources',
                  'rank hosts by alerts', 'which ip has the most alarms', 'most suspicious devices on the network', 'offenders leaderboard', 'which endpoints are causing trouble', 'biggest sources of alarms', 'most active suspicious addresses', 'machines to investigate first', 'heaviest hitters on the network'],
    'attack_types': ['what kinds of attacks did we see', 'which attack families were detected', 'breakdown of attack types',
                     'are these denial of service or probes', 'what categories of intrusion today', 'show attack distribution',
                     'types of threats observed', 'is it mostly dos traffic', 'which attack classes are most common', 'summarise the attacks by type', 'what sort of malicious traffic', 'group the alerts by category', 'scans or floods', 'which threat classes appear', 'mix of attack kinds', 'what is the attack profile'],
    'model_status': ['is the detector running', 'model health check', 'is the intrusion model online', 'status of the detection engine',
                     'is everything working', 'when was the model last updated', 'is the edge device alive', 'check system status',
                     'is the classifier up', 'detector heartbeat', 'are you connected to the sensor', 'is detection live', 'uptime of the detector', 'is the pipeline healthy', 'any errors on the device', 'is monitoring active'],
    'performance': ['how accurate is the model', 'what is the detection latency', 'how fast does it score a connection',
                    'what is the false alarm rate', 'model precision and recall', 'how good is the detector', 'what is the f1 score',
                    'how many milliseconds per sample', 'does it meet the latency budget', 'how reliable are the alerts', 'how often is it wrong', 'what is the recall', 'speed of inference', 'can we trust its alarms', 'error rate of the classifier', 'response time of the model'],
    'mark_false_positive': ['mark alert 5 as false positive', 'this alert is a false alarm', 'close alert 7 as benign',
                            'that was legitimate traffic', 'dismiss this alert', 'flag as false positive', 'alert 9 is not an attack',
                            'resolve this alert as harmless', 'it was a false alarm close it', 'report a false detection', 'that one is a mistake', 'whitelist this alert', 'benign activity close the ticket', 'label it as not malicious', 'wrong alarm please clear', 'false detection on that one'],
    'block_host': ['block host 2041', 'quarantine that machine', 'add this ip to the blocklist', 'cut off host 77',
                   'isolate the attacker', 'deny traffic from this source', 'block the suspicious host', 'firewall this address',
                   'disconnect host 1503', 'stop traffic from that device', 'ban that address', 'drop all packets from it', 'shut that machine out', 'cut the connection from that source', 'put it behind the firewall', 'blacklist the source'],
    'help': ['what can you do', 'help', 'list your commands', 'how do i use you', 'what questions can i ask',
             'show me the options', 'what are your features', 'i need help', 'menu', 'guide me', 'instructions please', 'what is supported', 'how should i talk to you', 'capabilities', 'usage', 'i am lost'],
}
TEST = {   # paraphrases written separately from the training phrases
    'open_alerts': ['what alerts are open now', 'show unresolved alarms', 'list the pending incidents', 'anything still open', 'current open detections', 'which alarms are active'],
    'explain_alert': ['why was alert 4 raised', 'explain this detection to me', 'what caused this flag', 'why is this connection suspicious', 'give the reason for alert 11', 'why did you flag host 300'],
    'alert_count': ['how many alarms so far today', 'count of detections this hour', 'total number of alerts', 'how many intrusions detected', 'alerts count for today', 'number of incidents this shift'],
    'top_hosts': ['top suspicious machines', 'which hosts raise most alarms', 'noisiest sources on the network', 'rank the most flagged hosts', 'which ip addresses are worst', 'hosts with most detections'],
    'attack_types': ['what attack types are we seeing', 'distribution of attack families', 'are they probes or dos attacks', 'kinds of intrusions detected', 'attack categories breakdown', 'what classes of attack occurred'],
    'model_status': ['is the model up and running', 'health of the detector', 'is the engine online', 'system status check', 'is the device working', 'detector status'],
    'performance': ['what is the model accuracy', 'scoring latency per connection', 'false alarm rate of the detector', 'precision of the model', 'is it fast enough', 'how good are its predictions'],
    'mark_false_positive': ['mark alert 2 as a false alarm', 'that alert is benign close it', 'false positive on alert 6', 'dismiss alert 8 as harmless', 'this was normal traffic', 'close this as false positive'],
    'block_host': ['block that ip address', 'quarantine host 812', 'put host 45 on the blocklist', 'isolate this device', 'deny the attacker', 'firewall host 2041'],
    'help': ['what are you able to do', 'show your commands', 'how does this work', 'help me please', 'which questions are supported', 'options menu'],
}
OOS_DEV = ['what is the weather tomorrow', 'tell me a joke', 'who won the cricket match', 'order a pizza', 'play some music',
           'what time is it in london', 'translate hello to french', 'book a meeting room', 'what is the capital of japan',
           'how do i bake bread', 'recommend a movie', 'set an alarm for 6 am', 'what is my salary', 'open youtube', 'convert dollars to rupees']
OOS_TEST = ['will it rain today', 'sing a song', 'who is the prime minister', 'buy me a coffee', 'what is the stock price of apple',
            'how tall is mount everest', 'remind me to call mom', 'best restaurants near me', 'how do i reset my email password',
            'what is two plus two', 'tell me about black holes', 'schedule leave for friday', 'turn on the lights', 'cricket score please',
            'write a poem about rain', 'how many calories in rice', 'who wrote hamlet', 'what day is it', 'call a taxi', 'news headlines']


class RetrievalBot:
    """TF-IDF over character 3-5-grams within word boundaries (robust to recognition errors and to word-level
    overlap such as "what is the" that out-of-scope requests share with intents); an utterance is mapped to the
    intent of its most similar training phrase; below `threshold` it is out of scope."""
    def __init__(self, intents, threshold=0.3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.phrases = [p for i in intents for p in intents[i]]
        self.labels = [i for i in intents for _ in intents[i]]
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit(self.phrases)
        self.M = self._vec(self.phrases)
        self.threshold = threshold

    def _vec(self, texts):
        from sklearn.preprocessing import normalize
        return normalize(self.char.transform(texts))

    def scores(self, texts):
        S = (self._vec(texts) @ self.M.T).toarray()
        j = S.argmax(1)
        return [self.labels[k] for k in j], S.max(1)

    def classify(self, text, threshold=None):
        lab, conf = self.scores([text])
        th = self.threshold if threshold is None else threshold
        return (lab[0] if conf[0] >= th else 'out_of_scope'), float(conf[0])


def corrupt(text, wer, rng, vocab):
    """Speech-recognition noise: each word is substituted / deleted / followed by an inserted word with
    total probability `wer` (substitution 60 %, deletion 25 %, insertion 15 %)."""
    out = []
    for w in text.split():
        u = rng.random()
        if u < wer * 0.60: out.append(_sound_alike(w, rng, vocab))
        elif u < wer * 0.85: continue
        elif u < wer: out += [w, rng.choice(vocab)]
        else: out.append(w)
    return ' '.join(out) if out else rng.choice(vocab)


def _sound_alike(w, rng, vocab):
    """Recognisers usually confuse similar-sounding words: pick a vocabulary word sharing the first letter
    or a one-character edit of the word; fall back to a random word."""
    cands = [v for v in vocab if v != w and v[0] == w[0] and abs(len(v) - len(w)) <= 2]
    if cands and rng.random() < 0.7: return rng.choice(cands)
    if len(w) > 3:
        i = rng.randrange(len(w)); return w[:i] + rng.choice('aeioustnrl') + w[i + 1:]
    return rng.choice(vocab)


def evaluate(bot, wer=0.0, threshold=None, seed=0, reps=5):
    """In-scope intent accuracy, out-of-scope detection rate and false-accept rate (OOS accepted as an intent)."""
    vocab = sorted({w for t in list(bot.phrases) + OOS_DEV for w in t.split()})
    rng = random.Random(seed)
    th = bot.threshold if threshold is None else threshold
    ins = [(u, i) for i in TEST for u in TEST[i]]
    correct = total = oos_ok = oos_n = 0
    pairs = []
    for _ in range(reps):
        utts = [corrupt(u, wer, rng, vocab) for u, _ in ins]
        lab, conf = bot.scores(utts)
        for (u, i), l, c in zip(ins, lab, conf):
            p = l if c >= th else 'out_of_scope'
            correct += p == i; total += 1; pairs.append((i, p))
        o = [corrupt(u, wer, rng, vocab) for u in OOS_TEST]
        lab, conf = bot.scores(o)
        for c in conf:
            oos_ok += c < th; oos_n += 1
    return dict(wer=wer, threshold=th, intent_acc=correct / total, oos_detect=oos_ok / oos_n,
                false_accept=1 - oos_ok / oos_n, pairs=pairs)


def choose_threshold(intents, grid=np.round(np.arange(0.10, 0.71, 0.02), 2), held=3):
    """Pick the threshold without touching the test sets: refit on all but the last `held` phrases of each
    intent, score those held phrases (in scope) and OOS_DEV (out of scope), maximise in-scope accuracy +
    out-of-scope rejection. The final bot is then refitted on all training phrases."""
    fit = {k: v[:-held] for k, v in intents.items()}
    dev = [(u, k) for k, v in intents.items() for u in v[-held:]]
    bot = RetrievalBot(fit)
    lab, conf = bot.scores([u for u, _ in dev])
    ok = np.array([l == k for l, (_, k) in zip(lab, dev)])
    _, conf_oos = bot.scores(OOS_DEV)
    best, curve = None, []
    for th in grid:
        acc = float(np.mean(ok & (conf >= th))); rej = float(np.mean(conf_oos < th))
        curve.append((float(th), acc, rej))
        if best is None or acc + rej > best[1]: best = (float(th), acc + rej, acc, rej)
    return best, curve
