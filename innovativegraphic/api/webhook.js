const crypto = require('crypto');

module.exports = async (req, res) => {
  // (1) Handle test requests
  if (req.method === "GET" || req.body?.type === "TEST") {
    return res.status(200).json({ status: "OK" });
  }

  // (2) Verify signature
  const secretKey = process.env.CASHFREE_SECRET_KEY; // ✅ Use correct env var name
  if (!secretKey) {
    console.error("❌ CASHFREE_SECRET_KEY is missing");
    return res.status(500).json({ error: "Server misconfiguration" });
  }

  const signature = req.headers['x-cf-signature'];
  const body = JSON.stringify(req.body);

  const expectedSignature = crypto
    .createHmac('sha256', secretKey) // ✅ Use the variable
    .update(body)
    .digest('base64');

  if (signature !== expectedSignature) {
    console.error("❌ Invalid signature");
    return res.status(401).json({ error: "Invalid signature" });
  }

  // (3) Process successful payment
  console.log("✅ Payment succeeded:", req.body.order_id);
  res.status(200).end();
};
