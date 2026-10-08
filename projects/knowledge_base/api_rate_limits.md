# API Rate Limits and Quotas

## 1. Overview
All client requests to the REST API are subject to rate limiting enforced per API key to ensure stability, prevent abuse, and guarantee quality of service across multi-tenant clusters.

## 2. Limits by Subscription Tier
Rate limits are calculated across a rolling 60-second window:
- **Basic Plan:** No programmatic API access permitted.
- **Pro Plan:** **300 requests per minute** per API key.
- **Enterprise Plan:** **1,200 requests per minute** per API key, with custom burst quotas available upon request.

## 3. Burst Capacity
- Pro tier keys permit short bursts up to 20% over quota (360 requests) for windows under 10 seconds.
- Enterprise keys permit burst spikes up to 1,500 requests per minute for windows under 30 seconds.

## 4. HTTP Headers and Status Codes
When an application makes a request, the following headers are returned with every response:
- `X-RateLimit-Limit`: Maximum requests permitted per minute in your current plan tier.
- `X-RateLimit-Remaining`: Number of requests remaining in the current 60-second window.
- `X-RateLimit-Reset`: Unix epoch timestamp indicating when the window resets.

When limits are exceeded:
- The server responds with status code `HTTP 429 Too Many Requests`.
- The response includes a `Retry-After` header indicating the number of seconds to wait before attempting the request again.

## 5. Recommended Throttling Strategy
Clients encountering HTTP 429 errors must implement exponential backoff with random jitter:
```python
delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
```
Do not immediately retry failed requests without backing off, as continued requests will extend the block window.
