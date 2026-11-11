from sqlalchemy import select

from src.db.exceptions import NotFoundError


class BaseRepository:
    """The basic CRUD functions. All the other repositories inherit from this class
    and set `model` to the table class they work on.

    The repositories never commit. That is done by session_scope() / get_session() at the end,
    so everything in one block is saved together.
    """

    model = None

    def __init__(self, session):
        self.session = session

    async def get(self, id):
        """Find one row by its primary key. Returns None if it does not exist.
        If the primary key has more than one column, give a tuple, e.g. get((run_id, epoch))."""
        return await self.session.get(self.model, id)

    async def get_all(self, **filters):
        """Get all rows that match the filters, e.g. get_all(run_id=3).
        Without filters you get the whole table, so be careful with big tables."""
        result = await self.session.execute(select(self.model).filter_by(**filters))
        return list(result.scalars().all())

    async def create(self, **values):
        """Insert a new row and return it."""
        row = self.model(**values)
        self.session.add(row)
        # flush sends the insert to the database so we get the generated id back
        await self.session.flush()
        return row

    async def update(self, id, **values):
        """Change some columns of a row, e.g. update(run_id, status="finished")."""
        row = await self.get(id)
        if row is None:
            raise NotFoundError(f"{self.model.__name__} with id {id} was not found")
        for column, value in values.items():
            setattr(row, column, value)
        await self.session.flush()
        return row

    async def delete(self, id):
        """Delete a row."""
        row = await self.get(id)
        if row is None:
            raise NotFoundError(f"{self.model.__name__} with id {id} was not found")
        await self.session.delete(row)
        await self.session.flush()
