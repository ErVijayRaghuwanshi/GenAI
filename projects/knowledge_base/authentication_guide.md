# Authentication and Token Management

## 1. Authentication Scheme
All API requests must be authenticated using HTTP Bearer Tokens passed inside the `Authorization` header:
```http
Authorization: Bearer YOUR_SECRET_API_KEY
```
Requests lacking this header or providing an invalid key return `HTTP 401 Unauthorized`.

## 2. Token Lifecycle & Expiration
- Standard API tokens remain active for a maximum lifetime of **90 days**.
- Security policy recommends rotating active production tokens every **60 days**.
- Tokens that exceed 90 days are automatically revoked by the system and result in error `ERR_AUTH_EXPIRED`.

## 3. Token Scopes
When generating a token in the Developer Settings portal, configure one or more granular scopes:
- `read:data`: Grants read access to search indexes, documents, and query pipelines.
- `write:data`: Grants permission to ingest, update, or purge document chunks.
- `admin`: Grants full administrative rights including team member management, billing settings, and token creation.

## 4. Immediate Token Revocation
If an API key is accidentally exposed in a public repository or client-side application:
1. Immediately navigate to **Settings > API Keys** in the web dashboard.
2. Click **Revoke Immediately** next to the compromised key ID.
3. Revocation takes effect globally across all edge clusters within **30 seconds**.
