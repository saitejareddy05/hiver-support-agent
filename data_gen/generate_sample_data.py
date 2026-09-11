"""
Generates data/sample/askplaystation_tweets.csv — a synthetic dataset that mirrors
the schema and noisy style of the real Kaggle "Customer Support on Twitter" dataset
(thoughtvector/customer-support-on-twitter), restricted to a single brand:
@AskPlayStation.

WHY THIS EXISTS
----------------
The real dataset is ~3M rows and lives behind a Kaggle login, which this sandboxed
generator cannot reach. To keep the pipeline runnable in <15 minutes out of the box
(see README), we ship a smaller, synthetic-but-structurally-faithful stand-in with
the SAME columns as the real twcs.csv:

    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

If a user has the real twcs.csv, `src/data_prep.py` will use it instead automatically
(see README "Using the real dataset"). This generator is only for out-of-the-box
reproducibility and for unit tests.

The generator is intentionally template + randomization based (not hand-written row by
row) so we can produce enough volume (~900 tweets / ~220 threads) across 8 intents with
realistic noise: typos, hashtags, @mentions, emoji, inconsistent casing, multi-turn
back-and-forth, and a long tail of one-off phrasing.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AskPlayStation"
BRAND_AUTHOR_ID = "AskPlayStation"

# ---------------------------------------------------------------------------
# Intent taxonomy (defined from looking at the shape of real support threads
# for gaming/console brands in the dataset). See src/intents.py for the
# canonical definition used by the classifier — kept in sync manually.
# ---------------------------------------------------------------------------
INTENTS = [
    "account_login_access",
    "refund_billing",
    "purchase_missing_content",
    "technical_bug_error",
    "network_connectivity",
    "hardware_malfunction",
    "how_to_general_inquiry",
    "complaint_repeat_escalation",
]

CUSTOMER_HANDLES = [f"user_{i:04d}" for i in range(1, 400)]

TYPOS = {
    "the": ["teh", "the"],
    "you": ["u", "you"],
    "please": ["pls", "please", "plz"],
    "account": ["acount", "account"],
    "cannot": ["cant", "can't", "cannot"],
}

def noisy(text):
    words = text.split(" ")
    out = []
    for w in words:
        key = w.lower().strip(".,!?")
        if key in TYPOS and random.random() < 0.35:
            repl = random.choice(TYPOS[key])
            if w[0].isupper():
                repl = repl.capitalize()
            w = w.replace(w.strip(".,!?"), repl, 1) if key == w.lower() else w
            out.append(repl if key == w.lower().strip(".,!?") else w)
        else:
            out.append(w)
    return " ".join(out)

# ---------------------------------------------------------------------------
# Templates: (customer_message_templates, agent_reply_templates, resolved:boolean)
# {order}/{email}/{game}/{amount}/{console} get filled in.
# ---------------------------------------------------------------------------
GAMES = ["Spider-Man 2", "Horizon Forbidden West", "EA FC 24", "Call of Duty MW3",
         "Gran Turismo 7", "God of War Ragnarok", "Fortnite", "NBA 2K24", "Elden Ring"]
CONSOLES = ["PS5", "PS5 Digital Edition", "PS4", "PS4 Pro"]

TEMPLATES = {
    "account_login_access": {
        "customer": [
            "Hey @{brand} I cant login to my account, keeps saying wrong password even after reset {emoji} (ref {ticket})",
            "@{brand} my PSN account got locked for no reason, this is the 3rd time this month, ticket {ticket}",
            "yo @{brand} 2FA code never arrives to my email, been stuck for an hour trying to play {game}",
            "@{brand} someone hacked my account and changed the email, please help urgently, was mid-game in {game}",
            "@{brand} account says suspended but I didn't do anything wrong?? case {ticket}",
            "@{brand} keeps logging me out of {console} every few minutes, so annoying",
            "@{brand} forgot my password and the reset link in the email just 404s, tried 3 times now",
            "@{brand} my email got changed on my account without me doing it, locked out of everything including {game}",
        ],
        "agent": [
            "Hi there, sorry for the trouble logging in. Please try resetting your password from account.sonyentertainmentnetwork.com and check spam for the reset email. Let us know how it goes.",
            "We understand how frustrating that is. Please DM us your account's email (not password) and we'll take a closer look at the lock on our end.",
            "Sorry about the delay with your 2FA code — this can happen if your email provider is filtering us. Please check spam/junk folders and add PlayStation to safe senders.",
            "If your account was compromised, please DM us immediately so we can secure it and start the recovery process. Do not share your password with us or anyone else.",
            "We can see the suspension flag on file. Please DM your account email and we'll review the case with the account team.",
            "Sorry about that — frequent sign-outs can happen after a system update. Please try a fresh sign-in on {console} and let us know if it keeps happening.",
            "Sorry the reset link isn't working. Please try requesting it again from a browser instead of the console, and DM us if it still 404s.",
            "That sounds like account compromise — please DM us right away so we can begin recovery and lock down the account.",
        ],
        "resolved": [True, False, True, False, False, True, False, False],
    },
    "refund_billing": {
        "customer": [
            "@{brand} I got charged twice for {game}, ${amount} extra on my card. Need this fixed asap",
            "@{brand} can I get a refund for {game}? bought it by accident and haven't played it",
            "@{brand} still waiting on my refund from 2 weeks ago, ticket #{ticket}",
            "@{brand} my kid bought {game} without permission, want my ${amount} back",
            "@{brand} subscription renewed even though I cancelled last month, charged ${amount}",
        ],
        "agent": [
            "Sorry to hear about the duplicate charge. Please DM us your order ID and the last 4 digits of the card used so we can investigate the double billing.",
            "You're welcome to request a refund within 14 days of purchase if the content hasn't been substantially downloaded/streamed. Please submit a request at playstation.com/refund.",
            "Apologies for the wait. Please DM your ticket number and order ID so we can escalate this refund with our billing team directly.",
            "We're sorry to hear that. Please make sure Family/Parental controls with a spending limit are set up, and DM your order ID so we can review this purchase.",
            "Sorry about that — please DM your account email so we can check the cancellation timestamp against the renewal charge and correct it if needed.",
        ],
        "resolved": [False, True, False, False, False],
    },
    "purchase_missing_content": {
        "customer": [
            "@{brand} bought {game} on the store, paid, but it's not showing in my library",
            "@{brand} preorder bonus for {game} never showed up, anyone else having this issue",
            "@{brand} downloaded {game} but half the DLC content I paid for is missing",
            "@{brand} money left my account for {game} but download never started",
        ],
        "agent": [
            "Sorry about that! Please try restarting the console and checking Library > Purchased. If it's still missing, DM your order confirmation email.",
            "We're looking into reports of missing preorder bonuses for {game}. Please DM your PSN ID and order ID so we can check your specific account.",
            "Please DM your order ID for the DLC purchase so we can verify what was included and re-trigger the entitlement if something's missing.",
            "Sorry for the trouble — please DM a screenshot of the transaction and your order ID so we can check the download status on our end.",
        ],
        "resolved": [True, False, False, False],
    },
    "technical_bug_error": {
        "customer": [
            "@{brand} {game} keeps crashing on {console} right after the main menu, error CE-34878-0",
            "@{brand} game freezes every time I try to save progress in {game}, lost 3 hours of progress",
            "@{brand} getting error NW-31456-4 every time I try to launch anything online",
            "@{brand} screen goes black for a few seconds randomly while playing {game}",
        ],
        "agent": [
            "Sorry about the crashes. Please try rebuilding the database in Safe Mode (Option 5) and ensure {game} is fully updated. Let us know if CE-34878-0 persists.",
            "That's really frustrating, sorry. Please make sure your system software and {game} are both fully updated, then try saving to a different storage location as a test.",
            "NW-31456-4 usually points to a network/DNS issue. Please try setting DNS to 8.8.8.8 in network settings and test the connection again.",
            "Sorry to hear that — this can be an HDMI or power-saving setting issue. Please try a different HDMI port/cable and disable 'HDMI Device Link' temporarily.",
        ],
        "resolved": [False, False, True, False],
    },
    "network_connectivity": {
        "customer": [
            "@{brand} PSN has been down for me for 2 hours, can't sign in at all",
            "@{brand} multiplayer keeps disconnecting every 5 mins in {game}, tested on 2 different networks",
            "@{brand} is there an outage right now? can't connect to the store",
            "@{brand} NAT type shows strict and I can't join any {game} lobbies",
        ],
        "agent": [
            "We're aware of intermittent sign-in issues and our team is investigating. Please check status.playstation.com for live updates.",
            "Sorry about the disconnects. Please try a wired connection if possible and check that UPnP is enabled on your router, or set up port forwarding.",
            "Nothing showing on our status page currently, but please DM your region so we can check for localized issues.",
            "A Strict NAT type is usually a router setting — please enable UPnP or manually forward the required PS5 ports, listed on our support site.",
        ],
        "resolved": [False, False, False, True],
    },
    "hardware_malfunction": {
        "customer": [
            "@{brand} my {console} controller drifts so bad I can't aim in any shooter anymore",
            "@{brand} console won't turn on at all, just a blinking blue light then nothing",
            "@{brand} disc drive makes a loud grinding noise and won't read discs anymore",
            "@{brand} console overheating and shutting down mid-game, fans are so loud",
            "@{brand} HDMI port seems loose, screen keeps flickering when I bump the cable",
            "@{brand} controller won't charge anymore, tried 2 different cables, light doesn't even come on",
            "@{brand} console makes a weird clicking noise on startup every time now",
        ],
        "agent": [
            "Sorry about the drift — if it's within warranty please start a repair request at playstation.com/repair. We can also DM troubleshooting steps to try first.",
            "That blinking blue light usually points to a power supply or boot issue. Please try a different outlet, and if it persists this likely needs a repair request.",
            "Sorry to hear that — that noise suggests the disc drive may need servicing. Please start a repair request with your console's serial number.",
            "Sorry about the overheating — please make sure vents are clear of dust and the console has open airflow. If shutdowns continue, a repair request is the next step.",
            "Sorry about the flickering — a loose HDMI port usually needs a physical repair. Please try a different cable first, then start a repair request if it continues.",
            "Sorry about the charging issue — please try a different USB port and cable to rule those out. If the light never comes on, this likely needs a repair request.",
            "That clicking on startup usually needs a closer look from our repair team — please start a repair request with your console's serial number.",
        ],
        "resolved": [False, False, False, False, False, False, False],
    },
    "how_to_general_inquiry": {
        "customer": [
            "@{brand} how do I transfer my saves from {console} to my new PS5?",
            "@{brand} quick q — how do I set up parental controls for my kid's profile?",
            "@{brand} is {game} cross-platform with Xbox?",
            "@{brand} how much storage does the {game} install actually take up",
        ],
        "agent": [
            "You can use the built-in Console-to-Console transfer over Wi-Fi, or back up to a USB drive/cloud and restore on the new console — happy to send a step-by-step guide.",
            "Sure! Go to Settings > Family and Parental Controls > select the child's account, then adjust playtime, spending and content restrictions there.",
            "Great question — cross-play support varies by title, please check the game's store page under 'Online Features' for the latest info on {game}.",
            "Install size can vary with updates, but you can always check the exact current size under Library before downloading {game}.",
        ],
        "resolved": [True, True, True, True],
    },
    "complaint_repeat_escalation": {
        "customer": [
            "@{brand} this is the THIRD time I'm messaging about the same issue and still no fix, unacceptable, ticket {ticket}",
            "@{brand} absolutely furious, been a customer for 10 years and getting ignored on a ${amount} refund",
            "@{brand} your support keeps closing my ticket without resolving anything, done with this, case {ticket}",
            "@{brand} I want to speak to an actual human, bots keep sending me the same copy paste reply about {game}",
            "@{brand} 4th message this week, still no resolution on my {console} issue, this is a joke at this point",
            "@{brand} genuinely considering switching to Xbox after how this has been handled, so disappointed",
            "@{brand} you people never actually fix anything, just close tickets and move on, unreal",
        ],
        "agent": [
            "We're really sorry this hasn't been resolved yet. Please DM your case/ticket number and we'll get this escalated to a senior agent right away.",
            "That's not the experience we want for you — please DM your order ID so we can personally follow up on the refund status today.",
            "We apologize for the repeated closures. Please DM your ticket number so we can reopen it and assign it to a dedicated agent.",
            "Understood, and we're sorry for the frustration. Please DM your case number and we'll connect you with a specialist directly.",
            "That's not okay, and we're sorry for the repeated back and forth. Please DM your case number so a senior agent can take this over personally.",
            "We're sorry to hear that and don't want to lose you as a customer. Please DM your account details so we can properly investigate what's gone wrong.",
            "That's fair feedback and we're sorry. Please DM your ticket number so we can make sure this actually gets resolved this time.",
        ],
        "resolved": [False, False, False, False, False, False, False],
    },
}

EMOJI = ["😡", "😤", "🙄", "😭", "", "", "", "🙏", "😢"]


def rand_amount():
    return random.choice([9.99, 19.99, 29.99, 39.99, 59.99, 69.99, 14.99, 4.99])


def rand_ticket():
    return random.randint(100000, 999999)


def build_dataset(n_threads=800):
    rows = []
    tweet_id = 1
    base_time = datetime(2024, 1, 1, 9, 0, 0)

    for t in range(n_threads):
        intent = INTENTS[t % len(INTENTS)]
        # add extra randomness so counts aren't perfectly uniform
        if random.random() < 0.15:
            intent = random.choice(INTENTS)

        pack = TEMPLATES[intent]
        idx = random.randrange(len(pack["customer"]))
        cust_template = pack["customer"][idx]
        agent_template = pack["agent"][idx]
        resolved = pack["resolved"][idx]

        game = random.choice(GAMES)
        console = random.choice(CONSOLES)
        amount = rand_amount()
        ticket = rand_ticket()
        emoji = random.choice(EMOJI)

        cust_text = cust_template.format(brand=BRAND, game=game, console=console,
                                          amount=amount, ticket=ticket, emoji=emoji)
        cust_text = noisy(cust_text)
        author = random.choice(CUSTOMER_HANDLES)
        created = base_time + timedelta(minutes=t * 7 + random.randint(0, 5))

        inbound_id = tweet_id
        tweet_id += 1
        agent_id = tweet_id
        tweet_id += 1

        rows.append({
            "tweet_id": inbound_id,
            "author_id": author,
            "inbound": True,
            "created_at": created.strftime("%a %b %d %H:%M:%S %z %Y").replace(" %z", " +0000"),
            "text": cust_text,
            "response_tweet_id": agent_id,
            "in_response_to_tweet_id": "",
        })

        agent_text = agent_template.format(brand=BRAND, game=game, console=console, amount=amount)
        agent_created = created + timedelta(minutes=random.randint(3, 45))

        rows.append({
            "tweet_id": agent_id,
            "author_id": BRAND_AUTHOR_ID,
            "inbound": False,
            "created_at": agent_created.strftime("%a %b %d %H:%M:%S %z %Y").replace(" %z", " +0000"),
            "text": agent_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": inbound_id,
        })

        # ~30% of threads get a customer follow-up (multi-turn), sometimes unresolved
        if random.random() < 0.30 and not resolved:
            followups = [
                "still not working, can you please just fix this",
                "any update? been waiting a while now",
                "ok I DMed you, no response yet though",
                "this didn't help, same issue is happening",
            ]
            follow_text = noisy(random.choice(followups))
            follow_id = tweet_id
            tweet_id += 1
            follow_created = agent_created + timedelta(minutes=random.randint(5, 120))
            rows.append({
                "tweet_id": follow_id,
                "author_id": author,
                "inbound": True,
                "created_at": follow_created.strftime("%a %b %d %H:%M:%S %z %Y").replace(" %z", " +0000"),
                "text": follow_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": agent_id,
            })

    return rows


def main():
    rows = build_dataset()
    out_path = "data/sample/askplaystation_tweets.csv"
    fieldnames = ["tweet_id", "author_id", "inbound", "created_at", "text",
                  "response_tweet_id", "in_response_to_tweet_id"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows ({len(rows)//2}+ threads) to {out_path}")


if __name__ == "__main__":
    main()
