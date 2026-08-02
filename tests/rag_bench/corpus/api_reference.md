# TaskFlow API Reference

**Version**: 2.4.1
**Base URL**: `https://api.taskflow.io/api/v2`
**Last Updated**: February 28, 2026

TaskFlow is a project management platform exposing a RESTful HTTP API. All requests and responses use JSON unless otherwise noted. This document describes the complete surface area of the TaskFlow v2 API, including authentication, resource endpoints, pagination, error handling, rate limiting, and webhook integration.

---

## Authentication

TaskFlow supports two authentication methods: API key authentication for server-to-server integrations, and OAuth 2.0 bearer tokens for user-facing applications.

### API Key Authentication

Include your API key in the `X-API-Key` request header for all requests. API keys are scoped to a single organization and do not expire unless explicitly revoked.

```bash
curl -X GET "https://api.taskflow.io/api/v2/users" \
  -H "X-API-Key: tf_live_Xk9mQzP4rBvN2sL8wYjE6dA1"
```

API keys can be created and revoked from the **Settings → Integrations → API Keys** panel in the TaskFlow dashboard. Each key is associated with a permission scope (see below). Store your API key securely; it grants full access to your organization's data within the declared scope.

### OAuth 2.0 Bearer Tokens

For applications acting on behalf of individual users, use OAuth 2.0. TaskFlow implements the Authorization Code flow with PKCE.

**Authorization Endpoint**: `https://auth.taskflow.io/oauth/authorize`
**Token Endpoint**: `https://auth.taskflow.io/oauth/token`

Include the bearer token in the `Authorization` header:

```bash
curl -X GET "https://api.taskflow.io/api/v2/users/me" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
```

OAuth tokens expire after **3600 seconds** (1 hour). Use the refresh token to obtain a new access token without re-prompting the user.

### Scopes

| Scope | Description |
|-------|-------------|
| `users:read` | Read user profiles and team membership |
| `users:write` | Create and modify user accounts |
| `orders:read` | View orders and order history |
| `orders:write` | Create, update, and cancel orders |
| `products:read` | Browse product catalog |
| `products:write` | Manage product listings |
| `webhooks:manage` | Register and delete webhook subscriptions |
| `admin` | Full administrative access (all scopes) |

---

## Users API

The Users API manages user accounts within your TaskFlow organization. Users represent individual people with authentication credentials and role assignments.

### List Users

Retrieve a paginated list of users in the organization.

**Endpoint**: `GET /api/v2/users`

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `offset` | integer | No | Number of records to skip. Default: `0` |
| `limit` | integer | No | Maximum records to return. Default: `25`, maximum: `100` |
| `role` | string | No | Filter by role: `admin`, `member`, `viewer`, `guest` |
| `status` | string | No | Filter by status: `active`, `inactive`, `pending` |
| `search` | string | No | Full-text search against name and email |
| `sort` | string | No | Sort field: `created_at`, `name`, `email`. Prefix with `-` for descending |

**Example Request**:

```bash
curl -X GET "https://api.taskflow.io/api/v2/users?offset=50&limit=25&role=member&sort=-created_at" \
  -H "X-API-Key: tf_live_Xk9mQzP4rBvN2sL8wYjE6dA1"
```

**Example Response** (`200 OK`):

```json
{
  "data": [
    {
      "id": "usr_7gH3kLmN9pQ2rS5t",
      "email": "priya.sharma@acme.com",
      "name": "Priya Sharma",
      "role": "member",
      "status": "active",
      "avatar_url": "https://cdn.taskflow.io/avatars/usr_7gH3kLmN9pQ2rS5t.png",
      "created_at": "2025-08-14T09:22:31Z",
      "last_active_at": "2026-02-27T16:44:05Z"
    }
  ],
  "meta": {
    "total": 347,
    "offset": 50,
    "limit": 25,
    "has_more": true
  }
}
```

### Create User

**Endpoint**: `POST /api/v2/users`

**Required Scope**: `users:write`

**Request Body**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `email` | string | Yes | User's email address (must be unique within organization) |
| `name` | string | Yes | Display name |
| `role` | string | Yes | Initial role assignment |
| `send_invite` | boolean | No | Send onboarding email. Default: `true` |
| `team_ids` | array[string] | No | Team IDs to add user to upon creation |

```bash
curl -X POST "https://api.taskflow.io/api/v2/users" \
  -H "X-API-Key: tf_live_Xk9mQzP4rBvN2sL8wYjE6dA1" \
  -H "Content-Type: application/json" \
  -d '{"email": "new.hire@acme.com", "name": "New Hire", "role": "member", "send_invite": true}'
```

**Response** (`201 Created`): Returns the newly created user object.

### Get User

**Endpoint**: `GET /api/v2/users/{id}`

Use the special identifier `me` to retrieve the authenticated user's own profile regardless of their user ID.

**Example**: `GET /api/v2/users/me`

**Response** (`200 OK`): Returns a single user object with additional fields: `preferences`, `notification_settings`, `two_factor_enabled`.

### Update User

**Endpoint**: `PATCH /api/v2/users/{id}`

Partial updates are supported — only include fields you wish to modify. All fields are optional.

**Updatable Fields**: `name`, `role`, `status`, `avatar_url`, `preferences`, `notification_settings`

---

## Orders API

Orders represent purchase transactions in the TaskFlow commerce module. An order progresses through a defined set of lifecycle states.

### Order Lifecycle

Orders follow a strict state machine:

```
pending → confirmed → processing → shipped → delivered
                ↓
            cancelled (from pending or confirmed only)
```

A `refunded` terminal state is available for delivered orders within the refund eligibility window (30 days post-delivery).

### List Orders

**Endpoint**: `GET /api/v2/orders`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `offset` | integer | No | Records to skip. Default: `0` |
| `limit` | integer | No | Records per page. Default: `25`, maximum: `100` |
| `status` | string | No | Filter by order status |
| `user_id` | string | No | Filter orders belonging to a specific user |
| `created_after` | string | No | ISO 8601 datetime. Returns orders created after this timestamp |
| `created_before` | string | No | ISO 8601 datetime. Returns orders created before this timestamp |
| `min_total` | number | No | Minimum order total in cents |
| `max_total` | number | No | Maximum order total in cents |

### Create Order

**Endpoint**: `POST /api/v2/orders`

The `line_items` array is required and must contain at least one item. Each line item references a product by `product_id` and specifies `quantity`. Total is computed server-side from current product pricing at the time of order creation.

### Update Order Status

**Endpoint**: `PATCH /api/v2/orders/{id}`

Only the `status` field may be updated via this endpoint. Status transitions must follow the order lifecycle diagram above; invalid transitions return `422 Unprocessable Entity`.

### Cancel Order

**Endpoint**: `DELETE /api/v2/orders/{id}`

Cancels an order if it is in `pending` or `confirmed` status. Returns `409 Conflict` if the order cannot be cancelled in its current state.

---

## Products API

The Products API exposes the TaskFlow product catalog. Products are the items that appear on order line items.

### List Products

**Endpoint**: `GET /api/v2/products`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `offset` | integer | No | Records to skip. Default: `0` |
| `limit` | integer | No | Records per page. Default: `25`, maximum: `100` |
| `category` | string | No | Filter by product category slug |
| `min_price` | integer | No | Minimum unit price in cents |
| `max_price` | integer | No | Maximum unit price in cents |
| `in_stock` | boolean | No | When `true`, returns only products with `stock_quantity > 0` |
| `search` | string | No | Full-text search against product name and description |
| `sort` | string | No | Sort field: `price`, `name`, `created_at`, `stock_quantity` |

### Create Product

**Endpoint**: `POST /api/v2/products`

**Required Scope**: `products:write`

**Request Body Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Product display name |
| `category_id` | string | Yes | Category identifier |
| `unit_price` | integer | Yes | Price in cents (e.g., `4999` for $49.99) |
| `sku` | string | Yes | Stock-keeping unit identifier (must be unique) |
| `description` | string | No | Markdown-formatted product description |
| `stock_quantity` | integer | No | Initial inventory count. Default: `0` |
| `attributes` | object | No | Key-value pairs for product specifications |

### Get Product

**Endpoint**: `GET /api/v2/products/{id}`

Accepts either the product's UUID or its `sku` as the `{id}` path parameter.

### Update Product

**Endpoint**: `PATCH /api/v2/products/{id}`

### Delete Product

**Endpoint**: `DELETE /api/v2/products/{id}`

Soft-deletes the product. Deleted products remain readable via `GET /api/v2/products/{id}` but are excluded from list results unless the query includes `include_deleted=true`. Orders referencing deleted products are unaffected.

---

## Pagination

All list endpoints in the TaskFlow API use **offset-based pagination**. The two controlling parameters are:

- **`offset`** — the number of records to skip before returning results (zero-indexed). To retrieve the second "page" of 25 results, set `offset=25`.
- **`limit`** — the maximum number of records to include in the response. The default value is `25`. The maximum permissible value is `100`. Requests with `limit` greater than `100` return a `400 Bad Request` error.

All list responses include a `meta` object:

```json
{
  "meta": {
    "total": 1284,
    "offset": 100,
    "limit": 25,
    "has_more": true
  }
}
```

- `total`: the total count of records matching the query, before applying `offset` and `limit`
- `offset`: the offset applied to this response
- `limit`: the limit applied to this response
- `has_more`: `true` when `offset + limit < total`

**Iterating Through All Records**: Increment `offset` by your `limit` value on each request until `has_more` is `false`. Avoid relying on `total` for loop termination, as records may be inserted or deleted between requests.

---

## Error Handling

All error responses use a consistent JSON structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request body contains invalid fields.",
    "details": [
      {
        "field": "email",
        "issue": "Must be a valid email address"
      },
      {
        "field": "role",
        "issue": "Must be one of: admin, member, viewer, guest"
      }
    ],
    "request_id": "req_5fH2mKpQ9rX1vN8t"
  }
}
```

### HTTP Status Codes

| Status | Code String | Meaning |
|--------|-------------|---------|
| `200` | — | Request succeeded |
| `201` | — | Resource created successfully |
| `204` | — | Request succeeded, no response body |
| `400` | `BAD_REQUEST` | Malformed request syntax |
| `401` | `UNAUTHORIZED` | Missing or invalid authentication credentials |
| `403` | `FORBIDDEN` | Authenticated but insufficient permissions for requested scope |
| `404` | `NOT_FOUND` | Requested resource does not exist |
| `409` | `CONFLICT` | Request conflicts with current resource state |
| `422` | `VALIDATION_ERROR` | Request body fails validation |
| `429` | `RATE_LIMITED` | Too many requests; see rate limiting section |
| `500` | `INTERNAL_ERROR` | Unexpected server error; contact support with `request_id` |
| `503` | `SERVICE_UNAVAILABLE` | Planned or unplanned maintenance; retry after the `Retry-After` header value |

The `request_id` field is present on all error responses and is required when contacting TaskFlow support.

---

## Rate Limiting

TaskFlow enforces per-organization rate limits on all API endpoints. Limits are applied on a rolling 60-second window.

| Plan Tier | Requests per Minute | Burst Allowance |
|-----------|---------------------|-----------------|
| Developer (free) | 100 | 20 |
| Standard | 1,000 | 200 |
| Professional | 2,500 | 500 |
| Enterprise | 5,000 | 1,000 |

Rate limit status is communicated via response headers included on every API response:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Total requests permitted per 60-second window |
| `X-RateLimit-Remaining` | Requests remaining in the current window |
| `X-RateLimit-Reset` | Unix timestamp when the current window resets |
| `Retry-After` | Seconds to wait before retrying (only present on `429` responses) |

When your application receives a `429 RATE_LIMITED` response, pause requests for the number of seconds specified by the `Retry-After` header. Implement exponential backoff with jitter for production systems — a naive retry loop risks cascading rate limit violations.

---

## Webhooks

Webhooks deliver real-time event notifications to an HTTP endpoint you control. TaskFlow sends a POST request to your registered endpoint URL whenever a subscribed event occurs.

### Registering a Webhook

**Endpoint**: `POST /api/v2/webhooks`

**Required Scope**: `webhooks:manage`

```json
{
  "url": "https://your-app.example.com/hooks/taskflow",
  "events": ["order.confirmed", "order.shipped", "user.created"],
  "secret": "your-32-character-minimum-secret-key",
  "active": true
}
```

**Supported Event Types**:

| Event | Triggered When |
|-------|----------------|
| `order.created` | A new order is placed |
| `order.confirmed` | Order transitions to confirmed state |
| `order.shipped` | Tracking information added |
| `order.delivered` | Delivery confirmed |
| `order.cancelled` | Order is cancelled |
| `order.refunded` | Refund processed |
| `user.created` | New user account created |
| `user.deactivated` | User account deactivated |
| `product.low_stock` | Product stock falls below threshold |

### Webhook Payload

```json
{
  "event_id": "evt_2Rk7mN9qP4sT1vL6",
  "event_type": "order.shipped",
  "created_at": "2026-02-14T13:05:22Z",
  "api_version": "2.4.1",
  "data": {
    "object": "order",
    "id": "ord_8Xc3nK5pM2rW9yH4",
    "status": "shipped",
    "tracking_number": "1Z999AA10123456784",
    "carrier": "UPS"
  }
}
```

### Signature Verification

Every webhook delivery includes a `X-TaskFlow-Signature` header containing an **HMAC-SHA256** signature. Verify this signature before processing the payload to ensure the request originated from TaskFlow and was not tampered with.

**Verification Algorithm**:

1. Retrieve the raw request body (before JSON parsing)
2. Compute `HMAC-SHA256(secret, raw_body)` where `secret` is the webhook secret you provided at registration
3. Compare the computed digest (hex-encoded) against the `X-TaskFlow-Signature` header value
4. Reject requests where the signatures do not match — use a constant-time comparison function to prevent timing attacks

**Example (Python)**:

```python
import hmac
import hashlib

def verify_signature(payload_body: bytes, secret: str, header_signature: str) -> bool:
    expected = hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header_signature)
```

### Retry Policy

If your endpoint returns any non-`2xx` HTTP response (or fails to respond within 10 seconds), TaskFlow will retry delivery up to **5 times** using exponential backoff:

| Attempt | Delay |
|---------|-------|
| 1 (initial) | Immediate |
| 2 | 30 seconds |
| 3 | 5 minutes |
| 4 | 30 minutes |
| 5 | 4 hours |

After all retry attempts are exhausted, the event is marked as failed. Failed events can be viewed in the dashboard under **Settings → Webhooks → Delivery Log** and manually replayed. Webhook endpoints that consistently fail are automatically deactivated after 100 consecutive failures.

Your endpoint must respond within 10 seconds. For long-running processing tasks, immediately return `200 OK` and handle the event asynchronously.
