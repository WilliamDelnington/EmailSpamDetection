adult_content_patterns = [
    r"(?i)\b(?:dating)\b",
    r"(?i)\b(?:one\snight\sstand)\b",
    r"(?i)\b(?:sex)\b",
    r"^(?=.*\bmature\b)(?=.*\bcontent\b).*$",
    r"(?i)\b(?:horny)\b",
    r"(?i)\bhot(?:\s\w)+?\s(girls|girl|woman|women|ladies)\b",
    r"(?i)\b(?:in\sbed)\b",
    r"(?i)\b(?:penis|pussy|dick)\b",
    r"(?i)\ber(?:\w)?ction(?:\w)?\b",
    r"(?i)\b(?:tits)\b",
    r"(?i)\b(?:boob|booby)\b",
    r"(?i)\b(?:cum|orgasm|orgasms)\b",
    r"(?i)\b(?:slutty|nude)\b",
    r"(?i)\bdaddy\b",
    r"(?i)\bsugar\s(mommy|baby)\b",
    r"(?i)\b(?:porn)\b",
    r"(?i)\b(?:sperm|sperms|sprms|sprm)\b",
    r"(?i)\b(?:wives)\b",
    r"(?i)\b(?:lonely\swomen|lonely\swoman)\b"
]

lottery_scam_content_patterns = [
    r"(?i)\b(?:lottery|prize|winner|won)\b",
    r"(?i)\b(?:congratulations|you\sare\sselected|you\sare\swinner)\b",
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
    r"(?i)\b(?:risk\sfree|no\sobligation|cheap|inexpensive|affordable)\b",
    r"(?i)\b(?:loan|credit|debt)\b",
    r"^(?=.*\b(low|no|less)\b)(?=.*\b(cost|costs|price|prices|priced)\b).*$",
    r"(?i)\b(?:bank|account|financial)\b",
    r"(?i)\b(?:money\sback|satisfaction\sguaranteed)\b",
    r"(?i)\b(?:wealth|rich|prosperity)\b",
    r"(?i)\b(?:house|car)\b"
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
    r"(?i)\b(?:guaranteed|approved|certified)\b",
    r"(?i)\b(?:business|opportunity|work\from\shome)\b",
    r"(?i)\b(?:earn\smoney|extra\scash)\b",
    r"(?i)\b(?:financial\sfreedom|financial\sfuture)\b",
    r"(?i)\b(?:services|products|solutions)\b",
    r"(?i)\b(?:internet\smarketing|online\sbusiness)\b",
    r"(?i)\b(?:company)\b",
    r"(?i)\b\$\b",
    r"(?i)\b(?:phone|mobile|call)\b",
    r"(?i)\b(?:prom)\b",
    r"(?i)\b(?:economical)\b",
    r"(?i)\b(?:pill)\b"
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
    r"(?i)\b(?:confirm)\b"
]

malware_content_patterns = [
    r"(?i)\b(?:download|install|click)\b",
    r"(?i)\b(?:free\ssoftware|free\sdownload)\b",
    r"(?i)\b(?:urgent|important\supdate)\b",
    r"(?i)\b(?:open\sthis\sattachment)\b",
    r"(?i)\b(?:risk\sfree|no\scost)\b",
    r"(?i)\b(?:software|softwares|program|application)\b",
    r"(?i)\b(?:virus)\b"
]

patterns = {
    "Adult Content": adult_content_patterns,
    "Lottery Scam": lottery_scam_content_patterns,
    "Financial Fraud": financial_scam_content_patterns,
    "Advertisement": advetise_content_patterns,
    "Phishing": phishing_content_patterns,
    "Malware": malware_content_patterns
}

url_patterns = (
    r"(?i)https?\s*:\s*\/\s*\/\s*" # http(s):// (protocol)
    r"(?:(?:[a-zA-Z0-9\-]\s*)+\.\s*)+(?:[a-zA-Z0-9\-]\s*)+" # domain name
    r"(?:\/\s*(?:[\w\.\-\/%=&\?]\s*)*)?" # path
)

date_patterns = (
    r"(?i)(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?\s*,?\s*\d{1,2}(?:st|nd|rd|th)?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4})" # date , dd mm yyyy"
    r"|(?:(?:\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})|(?:\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|feburary|march|april|june|july|august|september|october|december)\s*\d{2,4}))" # dd-mm-yyyy or dd Month yyyy
    r"|(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|feburary|march|april|june|july|august|september|october|december)\s*\d{1,2}\s*,?\s*\d{2,4})" # Month dd, yyyy
    r"|(?:\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|feburary|march|april|june|july|august|september|october|december))" # dd Month
    r"|(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|feburary|march|april|june|july|august|september|october|december)\s*\d{1,2})" # Month dd
)

time_pattern = (
    r"(?i)(?:\d{1,2}\s*:\s*\d{1,2}\s*:\s*\d{1,2})" #hh:mm:ss
    r"|(?:(?:\d{1,2}\s*:\d{2}\s*[AP]M)|(?:\d{1,2}\s*:\s*\d{2}))" # hh:mm AM/PM or hh:mm or hh AM/PM
    r"|(?:\d{1,2}\s*:\s*\d{2})" # hh:mm
)

phone_number_pattern = (
    r"(?:\+\d{1,3}\s*\(\d{3}\)\s*\d{3}\s*[-/]?\s*\d{4})"
    r"|(?:\d{10,11})"
    r"|(?:\+\d{1,3}\s*\d{1,2}\s*[-/]?\s*\d{3,4}\s*[-/]?\s*\d{4})"
    r"|(?:\+\d{1,3}\s*\d{9})"
    r"|(?:\+\d{2,5}\s*\d{5,8})"
    r"|(?:\(\d{1,3}\)\s*\d{3,4}\s*[-/]?\d{4})"
    r"|(?:\d{1}\s*\d{3}\s*\d{3}\s*\d{4})"
    r"|(?:\d{4,5}\s*[-/]?\s*\d{4,5})"
)

email_pattern = r'(?i)[a-zA-Z0-9._%+-]+\@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

spaced_out_email_pattern = r"[a-zA-Z0-9._%+-]+\s*@\s*(?:[a-zA-Z0-9-]+\s*\.\s*)+[a-zA-Z]{2,}"

money_pattern = (
    r"(?:(\u0024|\u00A2|\u00A3|\u00A5|USD|EUR|CAD|JPY|AUD|GBP|KRW)\s?\d+\s?\.\s?\d{1,3})"
    r"|(?:\d+\s?\.\s?\d{1,3}\s?(USD|EUR|CAD|JPY|AUD|GBP|KRW|\u0024|\u00A2|\u00A3|\u00A5))"
)