import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { AstryxPreview } from './AstryxPreview';

export const metadata: Metadata = {
  title: 'Astryx preview · Jruva',
  robots: { index: false, follow: false },
};

// Design comparison page for the Astryx migration. Hidden on production.
export default function AstryxPreviewPage() {
  if (process.env.VERCEL_ENV === 'production') notFound();
  return <AstryxPreview />;
}
