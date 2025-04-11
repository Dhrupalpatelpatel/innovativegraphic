const crypto = require('crypto');
// Assume a database model for orders
// const Order = require('../models/Order');

module.exports = async (req, res) => {
  // Handle test requests
  if (req.method === 'GET' || req.body?.type === 'TEST') {
    return res.status(200).json({ status: 'OK' });
  }

  try {
    // Verify signature
    const secretKey = process.env.CASHFREE_SECRET_KEY;
    if (!secretKey) {
      console.error('CASHFREE_SECRET_KEY missing');
      return res.status(500).json({ error: 'Server misconfiguration' });
    }

    const signature = req.headers['x-webhook-signature'];
    const rawBody = JSON.stringify(req.body);
    const expectedSignature = crypto
      .createHmac('sha256', secretKey)
      .update(rawBody)
      .digest('base64');

    if (signature !== expectedSignature) {
      console.error('Invalid webhook signature');
      return res.status(401).json({ error: 'Invalid signature' });
    }

    // Process webhook event
    const { data, event } = req.body;
    const orderId = data?.order?.order_id;

    switch (event) {
      case 'PAYMENT_SUCCESS':
        console.log(`Payment succeeded for order: ${orderId}`);
        // Update order status in database
        /*
        await Order.update(
          { status: 'completed', paymentStatus: data.payment.status, updatedAt: new Date() },
          { where: { orderId } }
        );
        */
        // TODO: Send confirmation email to customer
        break;

      case 'PAYMENT_FAILED':
        console.log(`Payment failed for order: ${orderId}`);
        // Update order status
        /*
        await Order.update(
          { status: 'failed', paymentStatus: data.payment.status, updatedAt: new Date() },
          { where: { orderId } }
        );
        */
        break;

      case 'REFUNDED':
        console.log(`Refund processed for order: ${orderId}`);
        // Update order status
        /*
        await Order.update(
          { status: 'refunded', paymentStatus: 'refunded', updatedAt: new Date() },
          { where: { orderId } }
        );
        */
        break;

      default:
        console.log(`Unhandled webhook event: ${event}`);
    }

    return res.status(200).json({ status: 'OK' });

  } catch (error) {
    console.error('Webhook error:', error.message);
    return res.status(500).json({ error: 'Webhook processing failed' });
  }
};