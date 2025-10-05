import {
  Links,
  Meta,
  MetaFunction,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";
import "./tailwind.css";
import "./index.css";
import React from "react";
import { Toaster } from "react-hot-toast";

export function Layout({ children }: { children: React.ReactNode }) {
  // Use a ref to track hydration to avoid conditional hooks/risky DOM manipulation
  const [isSsr, setIsSsr] = React.useState(true);
  React.useEffect(() => {
    // Mark as client-side after first render
    setIsSsr(false);
  }, []);

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        {/* Render portal targets only on client after hydration completes */}
        {!isSsr && <>
          <div id="modal-portal-exit" />
          <div id="root-outlet" />
        </>}
        <ScrollRestoration />
        <Scripts />
        <Toaster />
      </body>
    </html>
  );
}

export const meta: MetaFunction = () => [
  { title: "OpenHands" },
  { name: "description", content: "Let's Start Building!" },
];

export default function App() {
  return <Outlet />;
}
