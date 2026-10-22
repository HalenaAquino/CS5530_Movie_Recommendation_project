import { useEffect, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

async function api(path, options) {
  const res = await fetch(API + path, options);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

/* ---------- small helpers ---------- */

function BarList({ data, labelKey, valueKey }) {
  const max = Math.max(...data.map((d) => d[valueKey]), 1);
  return (
    <ul className="bars">
      {data.map((d) => (
        <li key={d[labelKey]}>
          <span className="bar-label" title={d[labelKey]}>{d[labelKey]}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${(d[valueKey] / max) * 100}%` }} />
          </span>
          <span className="bar-value">{d[valueKey]}</span>
        </li>
      ))}
    </ul>
  );
}

function Status({ loading, error }) {
  if (error) return <p className="error">{error}. Is the backend running on {API}?</p>;
  if (loading) return <p className="muted">Loading…</p>;
  return null;
}

function useFetch(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api(path).then(setData).catch((e) => setError(e.message));
  }, [path]);
  return { data, error, loading: !data && !error };
}

/* ---------- Data overview ---------- */

function Overview() {
  const { data, error, loading } = useFetch("/summary");
  if (!data) return <Status loading={loading} error={error} />;

  const ratingDist = Object.entries(data.rating_distribution).map(([r, n]) => ({ rating: r, count: n }));
  const tagList = Object.entries(data.top_tags).map(([tag, n]) => ({ tag, count: n }));

  return (
    <>
      <section className="stats">
        <div><strong>{data.n_users}</strong> users</div>
        <div><strong>{data.n_movies_rated}</strong> movies rated</div>
        <div><strong>{data.n_ratings.toLocaleString()}</strong> ratings</div>
        <div><strong>{data.n_tags.toLocaleString()}</strong> tags</div>
      </section>
      <p className="muted">
        Ratings span {data.time_range[0].slice(0, 10)} to {data.time_range[1].slice(0, 10)}.{" "}
        {(data.share_movies_under_5_ratings * 100).toFixed(0)}% of movies have fewer than 5 ratings.
        Median ratings per user: {data.ratings_per_user["50%"]}; per movie: {data.ratings_per_movie["50%"]}.
      </p>

      <div className="grid">
        <section>
          <h3>Rating distribution</h3>
          <BarList data={ratingDist} labelKey="rating" valueKey="count" />
        </section>
        <section>
          <h3>Most-rated movies</h3>
          <BarList data={data.top_movies} labelKey="title" valueKey="n_ratings" />
        </section>
        <section>
          <h3>Most common tags</h3>
          <BarList data={tagList} labelKey="tag" valueKey="count" />
        </section>
      </div>
    </>
  );
}

/* ---------- Recommend ---------- */

function Recommend() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [myRatings, setMyRatings] = useState([]);
  const [recs, setRecs] = useState(null);
  const [error, setError] = useState(null);

  async function search(e) {
    e.preventDefault();
    if (!query.trim()) return;
    try {
      setResults(await api(`/search?q=${encodeURIComponent(query)}`));
      setError(null);
    } catch (err) { setError(err.message); }
  }

  function addRating(movie) {
    setMyRatings((prev) =>
      prev.some((r) => r.movieId === movie.movieId) ? prev : [...prev, { ...movie, rating: 5 }]
    );
  }
  function setRating(id, rating) {
    setMyRatings((prev) => prev.map((r) => (r.movieId === id ? { ...r, rating } : r)));
  }
  function removeRating(id) {
    setMyRatings((prev) => prev.filter((r) => r.movieId !== id));
  }

  async function getRecs() {
    const body = {
      k: 10,
      ratings: myRatings.map((r) => ({ movieId: r.movieId, rating: r.rating })),
    };
    try {
      setRecs(await api("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }));
      setError(null);
    } catch (err) { setError(err.message); }
  }

  return (
    <div className="grid two">
      <section>
        <h3>Tell us what you like</h3>
        <form onSubmit={search} className="row">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search a title, e.g. Matrix" />
          <button type="submit">Search</button>
        </form>
        <ul className="list">
          {results.map((m) => (
            <li key={m.movieId}>
              <span>{m.title} <small>{m.genres}</small></span>
              <button className="ghost" onClick={() => addRating(m)}>Add</button>
            </li>
          ))}
        </ul>

        <h3>Your ratings</h3>
        {myRatings.length === 0 && <p className="muted">Search and add a few movies you have seen.</p>}
        <ul className="list">
          {myRatings.map((r) => (
            <li key={r.movieId}>
              <span>{r.title}</span>
              <span className="row">
                <select value={r.rating} onChange={(e) => setRating(r.movieId, Number(e.target.value))}>
                  {[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5].map((v) => <option key={v}>{v}</option>)}
                </select>
                <button className="ghost" onClick={() => removeRating(r.movieId)}>Remove</button>
              </span>
            </li>
          ))}
        </ul>

        <button disabled={myRatings.length === 0} onClick={getRecs}>Recommend movies</button>
        {error && <p className="error">{error}</p>}
      </section>

      <section>
        <h3>Recommendations</h3>
        {!recs && <p className="muted">Results appear here with the reasons behind each one.</p>}
        {recs && (
          <>
            <p className="muted">Based on {recs.n_liked_used} movie(s) rated 4 or higher.</p>
            <ol className="recs">
              {recs.recommendations.map((r) => (
                <li key={r.movieId}>
                  <strong>{r.title}</strong>
                  <small>{r.genres}</small>
                  {r.tags.length > 0 && <div className="tags">{r.tags.map((t) => <span key={t}>{t}</span>)}</div>}
                  {r.reasons.length > 0 && (
                    <div className="why">
                      Because you liked:{" "}
                      {r.reasons.map((x) => `${x.title} (${x.similarity})`).join(", ")}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          </>
        )}
      </section>
    </div>
  );
}

/* ---------- App shell ---------- */

const TABS = [
  ["recommend", "Recommend", Recommend],
];

export default function App() {
  const [tab, setTab] = useState("recommend");
  const Active = TABS.find((t) => t[0] === tab)[2];

  return (
    <main>
      <header>
        <h1>Movie recommender</h1>
        <nav>
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "tab active" : "tab"} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <Active />
    </main>
  );
}