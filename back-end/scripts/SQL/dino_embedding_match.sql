CREATE OR REPLACE FUNCTION match_image_dino_hq(
    person_id INTEGER,
    query_embedding vector(1536),
    match_threshold float,
    match_count integer
)
RETURNS TABLE (
    id bigint,
    path TEXT,
    stack bigint,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.id,
        i.path,
        i.stack,
        (1 - (i.image_embedding_hq <=> query_embedding))::FLOAT AS similarity
    FROM image i
    JOIN stack s ON i.stack = s.id
    JOIN person p ON s.person = p.id
    WHERE
        p.id = person_id
        AND (i.image_embedding_hq <=> query_embedding) < 1 - match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$
;