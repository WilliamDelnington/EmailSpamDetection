adult_content_patterns = [
    r"(?i)\bdating\b",
    r"(?i)\bone\snight\sstand\b",
    r"(?i)\bsex\b"
    r"^(?=.*\bmature\b)(?=.*\bcontent\b).*$",
    r"(?i)\bhorny\b"
    r"(?i)\bhot(?:\s\w+)?\s(girls|girl|woman|women|ladies)\b",
]

lottery_scam_content_patterns = [
    r"(?i)\b(?:lottery|prize|winner|won)\b",
]

financial_scam_content_patterns = [
    r"(?i)\b(?:money|cash|funds)\b",
    r"(?i)\b(?:investment|investor|invest)\b",
    r"(?i)\b(?:transfer)\b",
    r"(?i)\b(?:urgent|immediate|important)\b",
    r"(?i)\b(?:claim|verify|confirm)\b",
    r"(?i)\b(?:risk|guarantee|safe)\b",
    r"(?i)\b(?:limited|exclusive|special)\b",
    r"(?i)\b(?:offer|deal|discount)\b",
    r"(?i)\b(?:click|visit|apply)\b",
    r"(?i)\b(?:free|complimentary|bonus)\b",
    r"(?i)\b(?:click\shere|act\snow|call\snow)\b",
    r"(?i)\b(?:risk\sfree|no\sobligation|no\scost)\b",
]

advetise_content_patterns = [
    r"(?i)\b(?:buy|purchase|order)\b",
    r"(?i)\b(?:discount|sale|clearance)\b",
    r"(?i)\b(?:limited\stime|limited\soffer)\b",
    r"(?i)\b(?:free\s(?:trial|gift|sample))\b",
    r"(?i)\b(?:money\sback\sguarantee)\b",
    r"(?i)\b(?:best\sprice|lowest\sprice)\b",
    r"(?i)\b(?:save\sup\sto|save\sbig)\b",
    r"(?i)\b(?:call\stoday|act\stoday)\b",
    r"(?i)\b(?:get\spaid|make\smoney)\b",
]

phishing_content_patterns = [
    r"(?i)\b(?:login|account|password|access)\b",
    r"(?i)\b(?:update|verify|confirm)\b",
    r"(?i)\b(?:bank|credit\scard|financial)\b",
    r"(?i)\b(?:security|suspicious|unauthorized)\b",
    r"(?i)\b(?:click\shere|follow\sthis\slink)\b",
    r"(?i)\b(?:urgent|immediate\saction)\b",
    r"(?i)\b(?:suspended|deactivated|blocked)\b",
    r"(?i)\b(?:customer\sservice|support)\b",
]

malware_content_patterns = [
    r"(?i)\b(?:download|install|click)\b",
    r"(?i)\b(?:free\ssoftware|free\sdownload)\b",
    r"(?i)\b(?:urgent|important\supdate)\b",
    r"(?i)\b(?:open\sthis\sattachment)\b",
    r"(?i)\b(?:risk\sfree|no\scost)\b",
]

patterns = {
    "Adult Content": adult_content_patterns,
    "Lottery Scam": lottery_scam_content_patterns,
    "Financial Fraud": financial_scam_content_patterns,
    "Advertisement": advetise_content_patterns,
    "Phishing": phishing_content_patterns,
    "Malware": malware_content_patterns
}